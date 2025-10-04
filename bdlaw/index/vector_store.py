"""Vector store abstractions and Qdrant implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict
from typing import List, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from bdlaw.parser.structure import Norm
from bdlaw.settings.config import QdrantConfig


class VectorStore(ABC):
    @abstractmethod
    def upsert(self, norms: Sequence[Norm], vectors: Sequence[List[float]]) -> None:  # pragma: no cover
        ...

    @abstractmethod
    def search(
        self,
        query: List[float],
        k: int,
        filters: rest.Filter | None = None,
    ) -> List[rest.ScoredPoint]:  # pragma: no cover
        ...


class QdrantVectorStore(VectorStore):
    def __init__(self, config: QdrantConfig):
        self.client = QdrantClient(host=config.host, port=config.port, prefer_grpc=config.prefer_grpc)
        self.collection = config.collection
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        try:
            self.client.get_collection(self.collection)
        except Exception:  # pragma: no cover - depends on external service
            self.client.recreate_collection(
                collection_name=self.collection,
                vectors_config=rest.VectorParams(size=1536, distance=rest.Distance.COSINE),
            )

    def upsert(self, norms: Sequence[Norm], vectors: Sequence[List[float]]) -> None:
        payloads = [asdict(norm) for norm in norms]
        points = [
            rest.PointStruct(id=norm.gid, vector=vector, payload=payload)
            for norm, vector, payload in zip(norms, vectors, payloads)
        ]
        self.client.upsert(collection_name=self.collection, points=points)

    def search(
        self,
        query: List[float],
        k: int,
        filters: rest.Filter | None = None,
    ) -> List[rest.ScoredPoint]:
        return self.client.search(collection_name=self.collection, query_vector=query, limit=k, query_filter=filters)


__all__ = ["VectorStore", "QdrantVectorStore"]
