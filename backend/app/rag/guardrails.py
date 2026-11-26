"""Confidence guardrail: decide whether retrieved context is good enough to answer from.

Three bands on the best dense cosine similarity:

    score >= HIGH          -> ANSWER              (clearly on-topic, answer with citations)
    LOW <= score < HIGH    -> ask a VERIFIER LLM  ("do these passages actually answer this?")
    score < LOW            -> ESCALATE            (out of scope: never improvise, hand off)

Why two thresholds + a verifier instead of one threshold? A single cutoff forces a bad
trade-off: set it high and you escalate answerable questions; set it low and you answer
questions the KB doesn't cover. Similarity measures "topic overlap", not "contains the
answer". "Do you offer student discounts?" scores fairly close to the pricing doc but the
doc says nothing about students. The verifier only runs in the grey zone, so most
queries pay no extra latency or cost.

Overrides:
- An exact identifier match (e.g. the query mentions E-1042 and a chunk contains it) is
  treated as HIGH regardless of cosine score. Error codes are unambiguous.
- If the verifier itself fails (timeout, 429), we degrade to ANSWER_WITH_CAUTION: the
  agent may answer from the passages but must hedge and offer a human. Failing fully
  closed would turn every quota blip into an escalation.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum

from app.llm.base import LLMProvider
from app.llm.pricing import text_cost
from app.llm.resilience import classify
from app.observability.tracer import span
from app.rag.retriever import RetrievalResult

log = logging.getLogger(__name__)


class Band(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Decision(str, Enum):
    ANSWER = "answer"
    ANSWER_WITH_CAUTION = "answer_with_caution"
    ESCALATE = "escalate"


@dataclass
class Verdict:
    score: float
    band: Band
    decision: Decision
    reasons: list[str] = field(default_factory=list)
    verifier: dict | None = None

    def to_dict(self) -> dict:
        return {"score": round(self.score, 4), "band": self.band.value, "decision": self.decision.value,
                "reasons": self.reasons, "verifier": self.verifier}


VERIFIER_SYSTEM = (
    "You are a strict grounding checker for a customer support assistant. Decide whether the "
    "PASSAGES contain the information needed to answer the QUESTION. Do not use outside knowledge. "
    "'yes' = the passages directly answer it. 'partial' = they answer part of it or only a closely "
    "related case. 'no' = they do not address it."
)
VERIFIER_SCHEMA = {
    "type": "object",
    "properties": {
        "answerable": {"type": "string", "enum": ["yes", "partial", "no"]},
        "reason": {"type": "string", "description": "One short sentence."},
    },
    "required": ["answerable", "reason"],
}


class ConfidenceGate:
    def __init__(self, high: float, low: float, provider: LLMProvider | None, verifier_model: str,
                 verifier_thinking: str | None = "minimal", timeout_s: float = 8.0):
        assert low <= high, "confidence_low must be <= confidence_high"
        self.high, self.low = high, low
        self.provider = provider
        self.verifier_model = verifier_model
        self.verifier_thinking = verifier_thinking
        self.timeout_s = timeout_s

    def band_for(self, score: float) -> Band:
        if score >= self.high:
            return Band.HIGH
        if score >= self.low:
            return Band.MEDIUM
        return Band.LOW

    async def assess(self, question: str, result: RetrievalResult) -> Verdict:
        async with span("confidence_gate", "guardrail") as s:
            v = await self._assess(question, result)
            s.set(score=round(v.score, 4), band=v.band.value, decision=v.decision.value, reasons=v.reasons,
                  thresholds={"high": self.high, "low": self.low})
            return v

    async def _assess(self, question: str, result: RetrievalResult) -> Verdict:
        score = result.top_dense
        if not result.chunks:
            return Verdict(score, Band.LOW, Decision.ESCALATE, ["no passages retrieved"])
        if result.has_exact_match:
            return Verdict(score, Band.HIGH, Decision.ANSWER,
                           [f"exact identifier match ({', '.join(result.identifiers)})"])

        band = self.band_for(score)
        if band is Band.HIGH:
            return Verdict(score, band, Decision.ANSWER, [f"similarity {score:.2f} >= {self.high}"])
        if band is Band.LOW:
            return Verdict(score, band, Decision.ESCALATE, [f"similarity {score:.2f} < {self.low} (out of scope)"])

        # Grey zone: ask the verifier.
        reasons = [f"similarity {score:.2f} in grey zone [{self.low}, {self.high})"]
        if self.provider is None:
            return Verdict(score, band, Decision.ANSWER_WITH_CAUTION, reasons + ["no verifier configured"])
        verdict = await self._verify(question, result)
        if verdict is None:
            return Verdict(score, band, Decision.ANSWER_WITH_CAUTION, reasons + ["verifier unavailable"])
        decision = {"yes": Decision.ANSWER, "partial": Decision.ANSWER_WITH_CAUTION}.get(
            verdict["answerable"], Decision.ESCALATE)
        return Verdict(score, band, decision, reasons + [f"verifier: {verdict['answerable']} ({verdict['reason']})"],
                       verifier=verdict)

    async def _verify(self, question: str, result: RetrievalResult) -> dict | None:
        passages = "\n\n".join(f"[{c.chunk_id}] {c.breadcrumb}\n{c.text}" for c in result.chunks[:3])
        prompt = f"QUESTION:\n{question}\n\nPASSAGES:\n{passages}"
        t0 = time.perf_counter()
        async with span("answerability_verifier", "llm", model=self.verifier_model) as s:
            try:
                data, usage = await self.provider.generate_json(
                    model=self.verifier_model, system=VERIFIER_SYSTEM, prompt=prompt, schema=VERIFIER_SCHEMA,
                    thinking_level=self.verifier_thinking, timeout_s=self.timeout_s)
            except Exception as exc:  # noqa: BLE001 - degrade gracefully
                err = classify(exc)
                s.status = "error"
                s.set(error=err.to_dict())
                log.warning("verifier failed: %s", err)
                return None
            s.set(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
                  cost_usd=text_cost(self.verifier_model, usage.input_tokens, usage.output_tokens),
                  prompt=prompt, output=data)
        if data.get("answerable") not in ("yes", "partial", "no"):
            return None
        return {"answerable": data["answerable"], "reason": str(data.get("reason", ""))[:200],
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}
