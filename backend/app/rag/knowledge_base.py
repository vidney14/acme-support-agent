"""Knowledge base lifecycle: chunk -> embed -> index, with idempotent re-ingestion.

On startup we hash the KB markdown + chunking params + embedding model. If the vector
store already holds an index with the same hash, ingestion is skipped (fast restarts,
no wasted embedding quota). Any change to a doc, a chunking parameter or the embedding
model triggers a full re-index, so the index can never silently drift from the docs.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.rag.chunker import Chunk, chunk_directory
from app.rag.embeddings import Embedder
from app.rag.guardrails import ConfidenceGate, Verdict
from app.rag.retriever import HybridRetriever, RetrievalResult
from app.rag.store import VectorStore

log = logging.getLogger(__name__)


@dataclass
class KBStatus:
    ready: bool = False
    chunks: int = 0
    documents: int = 0
    ingested_at: float | None = None
    reused_index: bool = False
    error: str | None = None
    embedding_model: str = ""
    params: dict = field(default_factory=dict)


class KnowledgeBase:
    def __init__(self, kb_dir: Path, store: VectorStore, embedder: Embedder, gate: ConfidenceGate,
                 max_tokens: int, overlap_tokens: int, top_k: int):
        self.kb_dir = kb_dir
        self.store = store
        self.embedder = embedder
        self.gate = gate
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.top_k = top_k
        self.chunks: list[Chunk] = []
        self.retriever: HybridRetriever | None = None
        self.status = KBStatus(embedding_model=embedder.model,
                               params={"max_tokens": max_tokens, "overlap_tokens": overlap_tokens, "top_k": top_k})

    def _fingerprint(self) -> str:
        h = hashlib.sha256()
        for p in sorted(self.kb_dir.glob("*.md")):
            h.update(p.name.encode())
            h.update(p.read_bytes())
        h.update(f"{self.max_tokens}/{self.overlap_tokens}/{self.embedder.model}/{self.embedder.dim}".encode())
        return h.hexdigest()[:16]

    async def ingest(self, force: bool = False) -> KBStatus:
        try:
            self.chunks = chunk_directory(self.kb_dir, self.max_tokens, self.overlap_tokens)
            fp = self._fingerprint()
            meta = await self.store.get_meta()
            if not force and meta and meta.get("fingerprint") == fp:
                log.info("KB index up to date (%s), skipping embedding", fp)
                self.status.reused_index = True
            else:
                t0 = time.perf_counter()
                vectors = await self.embedder.embed_documents([c.embed_text for c in self.chunks])
                await self.store.recreate(self.embedder.dim)
                await self.store.upsert(self.chunks, vectors, meta={"fingerprint": fp, "model": self.embedder.model})
                log.info("Indexed %d chunks in %.1fs", len(self.chunks), time.perf_counter() - t0)
            self.retriever = HybridRetriever(self.store, self.embedder, self.chunks)
            self.status.ready = True
            self.status.error = None
            self.status.chunks = len(self.chunks)
            self.status.documents = len({c.doc_id for c in self.chunks})
            self.status.ingested_at = time.time()
        except Exception as exc:  # noqa: BLE001 - app must still boot; tools report KB unavailable
            log.exception("KB ingestion failed")
            self.status.ready = False
            self.status.error = f"{type(exc).__name__}: {exc}"[:300]
        return self.status

    async def search(self, query: str, tier: str | None) -> tuple[RetrievalResult, Verdict]:
        if not self.retriever:
            raise RuntimeError("knowledge base not ready")
        result = await self.retriever.retrieve(query, tier=tier, top_k=self.top_k)
        verdict = await self.gate.assess(query, result)
        return result, verdict
