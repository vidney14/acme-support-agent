"""Structure-aware markdown chunker for support policy documents.

Why not a fixed 500-character splitter? Support policies are *rules with conditions*
("refundable within 14 days, up to $50, unless past due"). A naive splitter happily
cuts a rule away from its condition, and then retrieval returns half a rule, which
is the worst possible input for a grounded answer. So this chunker:

1. **Splits on headings first** (H2/H3). A policy section is the natural unit of
   meaning, so a chunk never mixes "Pro refunds" with "Business refunds".
2. **Keeps blocks intact**: paragraphs, whole lists and tables are packed into a
   chunk as units; only an oversized block is split, and then at sentence boundaries.
   Tables are split by rows with the header row repeated.
3. **Adds overlap** (about 40 tokens of trailing sentences) between chunks of the same
   section, so a rule that straddles a boundary is fully present in at least one chunk.
4. **Prepends a breadcrumb** ("Refund Policy > Pro plan refunds > Pro annual") and the
   plans it applies to, *for embedding only*. Short chunks like "Pro annual charges can be
   refunded within 30 days" become unambiguous once the embedding knows the context.
5. **Inherits tier metadata**: a doc declares `tiers:` in front matter, and any section can
   narrow it with `<!-- tiers: pro -->`. Retrieval filters on this, so customers only see
   rules for their own plan.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

TIER_COMMENT = re.compile(r"<!--\s*tiers:\s*([a-z,\s]+)-->", re.IGNORECASE)
HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(*])")
TIER_LABELS = {"free": "Free", "pro": "Pro", "business": "Business", "enterprise": "Enterprise"}


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~0.75 words per token for English prose).

    Exact counts don't matter for chunk sizing: we only need consistent relative sizes,
    and avoiding a tokenizer call keeps ingestion fast and offline-friendly.
    """
    return max(1, round(len(re.findall(r"\S+", text)) * 1.33))


@dataclass
class Chunk:
    chunk_id: str               # "refund-policy#3", also used as the citation tag
    doc_id: str
    doc_title: str
    category: str
    section_path: list[str]     # ["Pro plan refunds", "Pro annual"]
    tiers: list[str]
    text: str                   # what the LLM and the UI see
    chunk_index: int
    token_count: int
    embed_text: str = ""        # breadcrumb + text: what the embedding model sees

    @property
    def breadcrumb(self) -> str:
        return " > ".join([self.doc_title, *self.section_path])

    def to_payload(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "category": self.category,
            "section_path": self.section_path,
            "breadcrumb": self.breadcrumb,
            "tiers": self.tiers,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
        }


@dataclass
class _Section:
    path: list[str]
    tiers: list[str]
    blocks: list[str] = field(default_factory=list)


def parse_front_matter(raw: str) -> tuple[dict, str]:
    if raw.startswith("---"):
        _, fm, body = raw.split("---", 2)
        return yaml.safe_load(fm) or {}, body
    return {}, raw


def _split_sections(body: str, doc_tiers: list[str]) -> list[_Section]:
    """Walk the markdown, tracking the heading stack and tier overrides."""
    sections: list[_Section] = []
    # stack entries: (level, title, tiers)
    stack: list[tuple[int, str, list[str]]] = []
    current = _Section(path=[], tiers=doc_tiers)
    buf: list[str] = []

    def flush_block() -> None:
        text = "\n".join(buf).strip()
        if text:
            current.blocks.append(text)
        buf.clear()

    for line in body.splitlines():
        h = HEADING.match(line)
        if h:
            flush_block()
            if current.blocks:
                sections.append(current)
            level, title = len(h.group(1)), h.group(2).strip()
            if level == 1:  # document title; the front matter title is used instead
                stack.clear()
                current = _Section(path=[], tiers=doc_tiers)
                continue
            while stack and stack[-1][0] >= level:
                stack.pop()
            inherited = stack[-1][2] if stack else doc_tiers
            stack.append((level, title, inherited))
            current = _Section(path=[s[1] for s in stack], tiers=inherited)
            continue

        t = TIER_COMMENT.search(line)
        if t:
            tiers = [x.strip().lower() for x in t.group(1).split(",") if x.strip()]
            current.tiers = tiers
            if stack:  # children of this heading inherit the override
                lvl, title, _ = stack[-1]
                stack[-1] = (lvl, title, tiers)
            continue

        if not line.strip():
            flush_block()
        else:
            buf.append(line)

    flush_block()
    if current.blocks:
        sections.append(current)
    return sections


def _split_oversized(block: str, max_tokens: int) -> list[str]:
    """Split a block that alone exceeds the budget: tables by rows, prose by sentences."""
    lines = block.splitlines()
    if len(lines) > 2 and lines[0].lstrip().startswith("|"):
        header, rows = lines[:2], lines[2:]
        parts, cur = [], []
        for row in rows:
            if cur and estimate_tokens("\n".join(header + cur + [row])) > max_tokens:
                parts.append("\n".join(header + cur))
                cur = []
            cur.append(row)
        if cur:
            parts.append("\n".join(header + cur))
        return parts

    if len(lines) > 1 and all(re.match(r"^\s*([-*]|\d+\.)\s", ln) for ln in lines):
        units = lines  # list: split between items, never inside one
        joiner = "\n"
    else:
        units = SENTENCE_END.split(block)
        joiner = " "
    parts, cur = [], []
    for u in units:
        if cur and estimate_tokens(joiner.join(cur + [u])) > max_tokens:
            parts.append(joiner.join(cur))
            cur = []
        cur.append(u)
    if cur:
        parts.append(joiner.join(cur))
    return parts


def _tail_overlap(text: str, overlap_tokens: int) -> str:
    """Last whole sentences of `text` totalling about `overlap_tokens`."""
    if overlap_tokens <= 0 or text.lstrip().startswith("|"):
        return ""
    sentences = SENTENCE_END.split(text)
    tail: list[str] = []
    for s in reversed(sentences):
        if estimate_tokens(" ".join([s, *tail])) > overlap_tokens:
            break
        tail.insert(0, s)
    return " ".join(tail)


def _pack(blocks: list[str], max_tokens: int, overlap_tokens: int) -> list[str]:
    units: list[str] = []
    for b in blocks:
        units.extend(_split_oversized(b, max_tokens) if estimate_tokens(b) > max_tokens else [b])

    chunks: list[str] = []
    cur: list[str] = []
    for u in units:
        if cur and estimate_tokens("\n\n".join(cur + [u])) > max_tokens:
            chunks.append("\n\n".join(cur))
            overlap = _tail_overlap(cur[-1], overlap_tokens)
            cur = [overlap] if overlap and overlap != cur[-1] else []
        cur.append(u)
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def _embed_header(doc_title: str, path: list[str], tiers: list[str]) -> str:
    plans = ", ".join(TIER_LABELS.get(t, t) for t in tiers)
    return f"{' > '.join([doc_title, *path])}\nApplies to: {plans} plan(s)"


def chunk_document(raw: str, max_tokens: int = 220, overlap_tokens: int = 40) -> list[Chunk]:
    meta, body = parse_front_matter(raw)
    doc_id = meta["doc_id"]
    title = meta.get("title", doc_id)
    category = meta.get("category", "general")
    doc_tiers = [t.lower() for t in meta.get("tiers", list(TIER_LABELS))]

    chunks: list[Chunk] = []
    for section in _split_sections(body, doc_tiers):
        for text in _pack(section.blocks, max_tokens, overlap_tokens):
            idx = len(chunks)
            chunk = Chunk(
                chunk_id=f"{doc_id}#{idx}",
                doc_id=doc_id,
                doc_title=title,
                category=category,
                section_path=section.path,
                tiers=section.tiers,
                text=text,
                chunk_index=idx,
                token_count=estimate_tokens(text),
            )
            chunk.embed_text = f"{_embed_header(title, section.path, section.tiers)}\n\n{text}"
            chunks.append(chunk)
    return chunks


def chunk_directory(kb_dir: Path, max_tokens: int = 220, overlap_tokens: int = 40) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        chunks.extend(chunk_document(path.read_text(encoding="utf-8"), max_tokens, overlap_tokens))
    return chunks
