from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Sequence

from .id_utils import chunk_sha
from .qdrant_client import SafeQdrantClient
from .settings import IngestRuntime

LOGGER = logging.getLogger(__name__)


@dataclass
class Chunk:
    doc_id: str
    text: str
    vector: Sequence[float]
    metadata: dict

    def to_payload(self, limits) -> dict:
        sha = chunk_sha(self.text)
        title = (self.metadata.get("title_text") or "")[: limits.title_max_chars]
        preview = (self.metadata.get("body_preview") or self.text[: limits.preview_max_chars])
        payload = dict(self.metadata)
        payload.update({
            "doc_id": self.doc_id,
            "sha256": sha,
            "title_text": title,
            "body_preview": preview[: limits.preview_max_chars],
        })
        return payload


class IngestService:
    def __init__(self, runtime: IngestRuntime | None = None) -> None:
        self.runtime = runtime or IngestRuntime()
        self.qdrant = SafeQdrantClient(self.runtime)

    def ingest(self, vector_dim: int, chunks: Iterable[Chunk]) -> str:
        collection = self.qdrant.create_physical_collection(vector_dim)
        self.qdrant.ensure_payload_indexes(collection)
        vectors: list[Sequence[float]] = []
        payloads: list[dict] = []
        limits = self.runtime.payload_limits
        for chunk in chunks:
            vectors.append(chunk.vector)
            payloads.append(chunk.to_payload(limits))
        if vectors:
            self.qdrant.upsert_points(collection, vectors, payloads)
        LOGGER.info("Ingested %s chunks into %s", len(vectors), collection)
        return collection


__all__ = ["Chunk", "IngestService"]
