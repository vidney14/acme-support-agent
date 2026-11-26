"""Hybrid retrieval: dense (Qdrant cosine) + sparse (BM25), fused with Reciprocal Rank Fusion.

Why hybrid? Dense embeddings are great at paraphrase ("I got billed twice" ~ "duplicate
charge") but weak on exact identifiers. Support traffic is full of identifiers: error
codes like E-1042, plan names, "P1". BM25 nails those. RRF merges the two rankings
without having to calibrate their incompatible score scales: each list votes
1/(k + rank) for its items.

Tier filtering happens in *both* retrievers, so a Pro customer can never be shown the
Business-only or Enterprise-only rules.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

import numpy as np
from rank_bm25 import BM25Okapi

from app.observability.tracer import span
from app.rag.chunker import Chunk
from app.rag.embeddings import Embedder, _features
from app.rag.store import VectorStore

IDENTIFIER = re.compile(r"\b([A-Z]{1,4}-\d{3,5})\b")
RRF_K = 60


@dataclass
class RetrievedChunk:
    chunk_id: str
    doc_title: str
    breadcrumb: str
    text: str
    tiers: list[str]
    dense_score: float | None
    bm25_score: float
    rrf_score: float
    exact_match: bool = False

    def to_dict(self) -> dict:
        return {"chunk_id": self.chunk_id, "doc_title": self.doc_title, "breadcrumb": self.breadcrumb,
                "text": self.text, "tiers": self.tiers,
                "dense_score": None if self.dense_score is None else round(self.dense_score, 4),
                "bm25_score": round(self.bm25_score, 3), "rrf_score": round(self.rrf_score, 5),
                "exact_match": self.exact_match}


@dataclass
class RetrievalResult:
    query: str
    tier: str | None
    chunks: list[RetrievedChunk] = field(default_factory=list)
    identifiers: list[str] = field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def top_dense(self) -> float:
        scores = [c.dense_score for c in self.chunks if c.dense_score is not None]
        return max(scores) if scores else 0.0

    @property
    def has_exact_match(self) -> bool:
        return any(c.exact_match for c in self.chunks)


class HybridRetriever:
    def __init__(self, store: VectorStore, embedder: Embedder, chunks: list[Chunk]):
        self.store = store
        self.embedder = embedder
        self.chunks = chunks
        self._by_id = {c.chunk_id: c for c in chunks}
        self._bm25 = BM25Okapi([_features(c.embed_text) for c in chunks]) if chunks else None

    def _bm25_ranked(self, query: str, tier: str | None, limit: int) -> list[tuple[str, float]]:
        if not self._bm25:
            return []
        scores = self._bm25.get_scores(_features(query))
        order = np.argsort(scores)[::-1]
        out = []
        for i in order:
            c = self.chunks[int(i)]
            if scores[i] <= 0:
                break
            if tier and tier not in c.tiers:
                continue
            out.append((c.chunk_id, float(scores[i])))
            if len(out) >= limit:
                break
        return out

    async def retrieve(self, query: str, tier: str | None, top_k: int = 4) -> RetrievalResult:
        t0 = time.perf_counter()
        result = RetrievalResult(query=query, tier=tier, identifiers=IDENTIFIER.findall(query.upper()))
        tiers = [tier] if tier else None

        async with span("embed_query", "embedding", model=self.embedder.model) as s:
            qvec = await self.embedder.embed_query(query)
            s.set(input_tokens=max(1, len(query) // 4))

        async with span("vector_search", "retrieval", tier=tier or "any") as s:
            dense = await self.store.search(qvec, limit=max(12, top_k * 3), tiers=tiers)
            sparse = self._bm25_ranked(query, tier, limit=max(12, top_k * 3))
            s.set(dense_hits=len(dense), bm25_hits=len(sparse))

        dense_score = {h.chunk_id: h.score for h in dense}
        bm25_score = dict(sparse)
        fused: dict[str, float] = {}
        for rank, h in enumerate(dense, start=1):
            fused[h.chunk_id] = fused.get(h.chunk_id, 0.0) + 1.0 / (RRF_K + rank)
        for rank, (cid, _) in enumerate(sparse, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank)

        candidates: list[RetrievedChunk] = []
        for cid, rrf in fused.items():
            c = self._by_id.get(cid)
            if c is None:
                continue
            # Identifiers often live in headings ("Error E-1042: path too long"), so match
            # against breadcrumb + body.
            haystack = f"{c.breadcrumb}\n{c.text}".upper()
            exact = any(ident in haystack for ident in result.identifiers)
            candidates.append(RetrievedChunk(
                chunk_id=cid, doc_title=c.doc_title, breadcrumb=c.breadcrumb, text=c.text, tiers=c.tiers,
                dense_score=dense_score.get(cid), bm25_score=bm25_score.get(cid, 0.0), rrf_score=rrf,
                exact_match=exact))
        # Exact identifier matches first, then by fused rank.
        candidates.sort(key=lambda r: (r.exact_match, r.rrf_score), reverse=True)
        result.chunks = candidates[:top_k]
        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result
