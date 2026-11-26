"""Embedding providers.

`GeminiEmbedder` is used in production. `HashingEmbedder` is a deterministic, offline
bag-of-words embedder used by the test suite (and the `fake` provider) so the whole
pipeline can be exercised without network access or an API key.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from typing import Protocol

import numpy as np

from app.llm.resilience import RetryPolicy, with_retries


class Embedder(Protocol):
    dim: int
    model: str

    async def embed_documents(self, texts: list[str], titles: list[str] | None = None) -> list[list[float]]: ...
    async def embed_query(self, text: str) -> list[float]: ...


def _normalize(vec: list[float]) -> list[float]:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr)) or 1.0
    return (arr / norm).tolist()


class GeminiEmbedder:
    """gemini-embedding-001 with asymmetric task types.

    Documents are embedded with RETRIEVAL_DOCUMENT and queries with RETRIEVAL_QUERY:
    the model is trained so a short question lands near the passage that answers it.
    Output is truncated to `dim` (Matryoshka representation) and re-normalised, because
    truncated vectors are no longer unit-length.
    """

    BATCH = 64

    def __init__(self, client, model: str, dim: int):
        self.client = client
        self.model = model
        self.dim = dim
        self.retry = RetryPolicy(max_retries=3, base_delay_s=1.0)

    async def _embed(self, texts: list[str], task_type: str, titles: list[str] | None = None) -> list[list[float]]:
        from google.genai import types

        async def call(batch: list[str]):
            resp = await self.client.aio.models.embed_content(
                model=self.model,
                contents=batch,
                config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dim),
            )
            return [_normalize(e.values) for e in resp.embeddings]

        out: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH):
            batch = texts[i : i + self.BATCH]
            out.extend(await with_retries(lambda b=batch: call(b), self.retry, label="embed"))
        return out

    async def embed_documents(self, texts: list[str], titles: list[str] | None = None) -> list[list[float]]:
        return await self._embed(texts, "RETRIEVAL_DOCUMENT", titles)

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([text], "RETRIEVAL_QUERY"))[0]


_STOP = set(
    "a an the and or of to in on for with is are be can do does i my me we our you your it this that "
    "what how when why who which there their at by from as if not no".split()
)


def _features(text: str) -> list[str]:
    words = [w for w in re.findall(r"[a-z0-9][a-z0-9\-]*", text.lower()) if w not in _STOP]
    stems = [w[:6] for w in words]  # crude stemming: refund/refunds/refunded -> refund
    return stems + [f"{a}_{b}" for a, b in zip(stems, stems[1:])]


class HashingEmbedder:
    """Feature-hashed TF vectors. Not semantic, but deterministic and good enough to test
    ranking, filtering and threshold logic offline."""

    model = "hashing-offline"

    def __init__(self, dim: int = 768):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for f in _features(text):
            h = int(hashlib.md5(f.encode()).hexdigest(), 16)
            v[h % self.dim] += 1.0 if (h >> 8) & 1 else -1.0
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]

    async def embed_documents(self, texts: list[str], titles: list[str] | None = None) -> list[list[float]]:
        await asyncio.sleep(0)
        return [self._vec(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        await asyncio.sleep(0)
        return self._vec(text)
