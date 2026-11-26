"""Qdrant vector store wrapper.

Runs Qdrant *embedded* (in-process, in-memory or on a local path) by default, so the
demo deploys as a single container with no extra infrastructure. Point QDRANT_URL at
Qdrant Cloud and nothing else changes: same client, same queries.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient, models

from app.rag.chunker import Chunk


@dataclass
class DenseHit:
    chunk_id: str
    score: float      # cosine similarity in [-1, 1]
    payload: dict


class VectorStore:
    def __init__(self, collection: str, url: str = "", api_key: str = "", path: str = ""):
        self.collection = collection
        self.remote = bool(url)
        if url:
            self.client = AsyncQdrantClient(url=url, api_key=api_key or None)
        elif path:
            self.client = AsyncQdrantClient(path=path)
        else:
            self.client = AsyncQdrantClient(location=":memory:")

    async def count(self) -> int:
        if not await self.client.collection_exists(self.collection):
            return 0
        return (await self.client.count(self.collection)).count

    async def get_meta(self) -> dict | None:
        """Ingestion metadata (KB hash, embedding model) stored as a sentinel point."""
        if not await self.client.collection_exists(self.collection):
            return None
        pts = await self.client.retrieve(self.collection, ids=[_point_id("__meta__")], with_payload=True)
        return pts[0].payload if pts else None

    async def recreate(self, dim: int) -> None:
        if await self.client.collection_exists(self.collection):
            await self.client.delete_collection(self.collection)
        await self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )
        if self.remote:
            # Payload index makes the tier filter cheap on a large KB (server Qdrant only;
            # embedded Qdrant filters by brute force, which is fine at this size).
            await self.client.create_payload_index(
                self.collection, field_name="tiers", field_schema=models.PayloadSchemaType.KEYWORD
            )

    async def upsert(self, chunks: list[Chunk], vectors: list[list[float]], meta: dict) -> None:
        points = [
            models.PointStruct(id=_point_id(c.chunk_id), vector=v, payload=c.to_payload())
            for c, v in zip(chunks, vectors)
        ]
        dim = len(vectors[0])
        # Sentinel point carrying ingestion metadata (never matches a tier filter).
        points.append(models.PointStruct(id=_point_id("__meta__"), vector=[1.0] + [0.0] * (dim - 1),
                                         payload={"__meta__": True, "tiers": [], **meta}))
        await self.client.upsert(self.collection, points=points, wait=True)

    async def search(self, vector: list[float], limit: int, tiers: list[str] | None) -> list[DenseHit]:
        must = [models.FieldCondition(key="tiers", match=models.MatchAny(any=tiers))] if tiers else []
        must_not = [models.FieldCondition(key="__meta__", match=models.MatchValue(value=True))]
        res = await self.client.query_points(
            self.collection,
            query=vector,
            limit=limit,
            with_payload=True,
            query_filter=models.Filter(must=must, must_not=must_not),
        )
        return [DenseHit(p.payload["chunk_id"], float(p.score), p.payload) for p in res.points]

    async def close(self) -> None:
        await self.client.close()


def _point_id(chunk_id: str) -> str:
    # Qdrant ids must be ints or UUIDs; derive a stable UUID from the chunk id.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"acme-kb/{chunk_id}"))
