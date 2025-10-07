from __future__ import annotations

import logging
from typing import Iterable, List

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from .chunker import Chunk
from .settings import AppSettings

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self._client = QdrantClient(
            url=settings.qdrant.url,
            api_key=settings.qdrant.api_key or None,
        )

    @property
    def collection_name(self) -> str:
        return self.settings.qdrant.collection

    def ensure_collection(self, vector_size: int, recreate: bool = False) -> None:
        collection = self.collection_name
        if recreate:
            try:
                self._client.delete_collection(collection)
            except Exception:
                logger.debug("Collection %s did not exist before recreation", collection)
        collections = {info.name for info in self._client.get_collections().collections}
        if collection not in collections:
            logger.info("Creating collection %s", collection)
            self._client.create_collection(
                collection_name=collection,
                vectors_config=rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE),
            )

    def upsert_chunks(self, chunks: Iterable[Chunk], vectors: List[List[float]]) -> None:
        payloads = []
        points = []
        for chunk, vector in zip(chunks, vectors):
            payload = {
                "doc_id": chunk.doc_id,
                "path": chunk.path,
                "page": chunk.page,
                "chunk_index": chunk.chunk_index,
                "offset": chunk.offset,
                "sha": chunk.sha,
                "text": chunk.text,
            }
            payloads.append(payload)
            points.append(
                rest.PointStruct(
                    id=f"{chunk.sha}_{chunk.chunk_index}",
                    vector=vector,
                    payload=payload,
                )
            )
        if not points:
            return
        self._client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector: List[float], top_k: int) -> List[rest.ScoredPoint]:
        return self._client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )

    def test_connection(self) -> bool:
        try:
            status = self._client.get_collection(self.collection_name)
            return status is not None
        except Exception:
            return False
