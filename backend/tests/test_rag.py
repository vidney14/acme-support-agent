"""Chunking, tier-filtered hybrid retrieval and the confidence gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import DATA_DIR
from app.llm.fake import ScriptedProvider
from app.llm.resilience import ErrorKind, ProviderError
from app.rag.chunker import chunk_directory, chunk_document, estimate_tokens
from app.rag.guardrails import Band, ConfidenceGate, Decision
from app.rag.retriever import RetrievalResult, RetrievedChunk

KB = DATA_DIR / "kb"


# ---------------------------------------------------------------- chunker
def test_section_tier_overrides_are_inherited_by_subsections():
    chunks = [c for c in chunk_directory(KB) if c.doc_id == "refund-policy"]
    by_path = {" > ".join(c.section_path): c for c in chunks}
    assert by_path["Pro plan refunds > Pro annual"].tiers == ["pro"]
    assert by_path["Business plan refunds > Seat reductions"].tiers == ["business"]
    assert set(by_path["Rules that apply to every refund"].tiers) == {"free", "pro", "business", "enterprise"}


def test_embed_text_carries_breadcrumb_and_plan_but_display_text_does_not():
    c = next(c for c in chunk_directory(KB) if c.section_path[-1:] == ["Pro annual"])
    assert c.embed_text.startswith("Refund Policy > Pro plan refunds > Pro annual\nApplies to: Pro")
    assert "Applies to" not in c.text


def test_chunks_respect_budget_and_tables_repeat_header_when_split():
    raw = (KB / "enterprise-sla.md").read_text()
    chunks = chunk_document(raw, max_tokens=45, overlap_tokens=15)
    tables = [c for c in chunks if c.text.startswith("| Severity")]
    assert len(tables) >= 2, "severity table should be split into several chunks"
    assert all(c.text.splitlines()[1].startswith("|---") for c in tables)
    # only the document preamble (text before the first heading) has no section path
    assert all(c.section_path or c.chunk_index == 0 for c in chunks)


def test_overlap_carries_trailing_sentence_into_next_chunk():
    raw = "---\ndoc_id: t\ntitle: T\n---\n## S\n" + " ".join(
        f"Sentence number {i} explains a rule in detail." for i in range(30))
    chunks = chunk_document(raw, max_tokens=60, overlap_tokens=20)
    assert len(chunks) >= 3
    last_sentence = chunks[0].text.split(". ")[-1].rstrip(".")
    assert last_sentence in chunks[1].text


def test_whole_kb_chunks_are_reasonably_sized():
    chunks = chunk_directory(KB)
    assert len({c.doc_id for c in chunks}) == 10
    assert max(estimate_tokens(c.text) for c in chunks) <= 260


# ---------------------------------------------------------------- retrieval
async def test_tier_filter_hides_other_plans_rules(container):
    res, _ = await container.kb.search("What service credits do I get for downtime?", "pro")
    assert all("pro" in c.tiers for c in res.chunks)
    assert not any(c.chunk_id.startswith("enterprise-sla") for c in res.chunks)
    res, _ = await container.kb.search("What service credits do I get for downtime?", "enterprise")
    assert any(c.chunk_id.startswith("enterprise-sla") for c in res.chunks)


async def test_exact_error_code_ranks_first_and_overrides_gate(container):
    res, verdict = await container.kb.search("my sync shows E-3003", "free")
    assert "E-3003" in res.chunks[0].breadcrumb
    assert verdict.decision is Decision.ANSWER
    assert "exact identifier" in verdict.reasons[0]


async def test_out_of_scope_question_escalates(container):
    _, verdict = await container.kb.search("Do you offer a student discount?", "pro")
    assert verdict.decision is Decision.ESCALATE


# ---------------------------------------------------------------- gate bands
def _result(score: float) -> RetrievalResult:
    r = RetrievalResult(query="q", tier="pro")
    r.chunks = [RetrievedChunk("d#0", "D", "D > S", "text", ["pro"], score, 1.0, 0.03)]
    return r


@pytest.mark.parametrize("score,verifier,expected,band", [
    (0.80, None, Decision.ANSWER, Band.HIGH),
    (0.40, None, Decision.ESCALATE, Band.LOW),
    (0.60, {"answerable": "yes", "reason": "r"}, Decision.ANSWER, Band.MEDIUM),
    (0.60, {"answerable": "partial", "reason": "r"}, Decision.ANSWER_WITH_CAUTION, Band.MEDIUM),
    (0.60, {"answerable": "no", "reason": "r"}, Decision.ESCALATE, Band.MEDIUM),
    (0.60, ProviderError(ErrorKind.RATE_LIMIT, "429", True), Decision.ANSWER_WITH_CAUTION, Band.MEDIUM),
])
async def test_confidence_bands_and_verifier(score, verifier, expected, band):
    provider = ScriptedProvider([], [verifier] if verifier is not None else [])
    gate = ConfidenceGate(high=0.72, low=0.55, provider=provider, verifier_model="v")
    v = await gate.assess("question", _result(score))
    assert v.band is band and v.decision is expected


def test_kb_files_exist():
    assert len(list(Path(KB).glob("*.md"))) == 10
