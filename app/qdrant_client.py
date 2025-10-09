from __future__ import annotations

import logging
import uuid
from typing import Iterable, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from .chunker import Chunk
from .settings import AppSettings

logger = logging.getLogger(__name__)

_EMBEDDING_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
}


class QdrantVectorStore:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        url = (settings.qdrant.url or "").strip()
        api_key = (settings.qdrant.api_key or "").strip()

        if not url:
            raise ValueError("Qdrant URL must be configured")

        client_kwargs: dict[str, object] = {
            "url": url,
            "timeout": settings.qdrant.timeout_seconds,
        }

        self._endpoint_url = url

        if url.startswith("https://"):
            if not api_key:
                raise ValueError("Qdrant API key is required for HTTPS endpoints")
            client_kwargs["api_key"] = api_key
        elif url.startswith("http://"):
            if "localhost" not in url and "127.0.0.1" not in url and api_key:
                client_kwargs["api_key"] = api_key
        else:
            raise ValueError(f"Unsupported Qdrant URL: {url}")

        logger.info(
            "Using Qdrant endpoint: %s (api_key=%s)",
            url,
            _mask_api_key(client_kwargs.get("api_key")),
        )

        self._client = QdrantClient(**client_kwargs)

    @property
    def collection_name(self) -> str:
        return self.settings.qdrant.collection

    @property
    def endpoint_url(self) -> str:
        return self._endpoint_url

    def ensure_collection(self, embedding_model: str, recreate: bool = False) -> None:
        collection = self.collection_name
        vector_size = _vector_size_for_model(embedding_model)
        vectors_config = rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE)

        if recreate:
            logger.info(
                "Recreating collection %s with dimension %d", collection, vector_size
            )
            self._client.recreate_collection(
                collection_name=collection,
                vectors_config=vectors_config,
            )
            return

        try:
            info = self._client.get_collection(collection_name=collection)
        except Exception:
            info = None

        current_size: Optional[int] = None
        if info and getattr(info, "config", None):
            params = getattr(info.config, "params", None)
            vectors = getattr(params, "vectors", None)
            if hasattr(vectors, "size"):
                current_size = getattr(vectors, "size")
            elif isinstance(vectors, dict):
                size = vectors.get("size")
                if isinstance(size, int):
                    current_size = size

        if current_size == vector_size:
            logger.info(
                "Collection %s already matches dimension %d", collection, vector_size
            )
            return

        if current_size is not None and current_size != vector_size:
            logger.warning(
                "Collection %s dimension %s differs from %s; recreating",
                collection,
                current_size,
                vector_size,
            )
        else:
            logger.info(
                "Collection %s missing; creating with dimension %d",
                collection,
                vector_size,
            )

        self._client.recreate_collection(
            collection_name=collection,
            vectors_config=vectors_config,
        )

    def upsert_chunks(
        self,
        chunks: Iterable[Chunk],
        vectors: List[List[float]],
        *,
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        chunk_list = list(chunks)
        if len(chunk_list) != len(vectors):
            raise ValueError(
                f"Chunk/vector length mismatch: received {len(chunk_list)} chunks "
                f"and {len(vectors)} vectors"
            )

        expected_dim = _vector_size_for_model(embedding_model)
        if vectors:
            actual_dim = len(vectors[0])
            if actual_dim != expected_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {expected_dim} "
                    f"(model={embedding_model}) but received {actual_dim}"
                )

        batch_size = max(1, self.settings.qdrant.upsert_batch_size)
        points: List[rest.PointStruct] = []
        for chunk, vector in zip(chunk_list, vectors):
            payload = {
                "doc_id": chunk.doc_id,
                "path": chunk.path,
                "page": chunk.page,
                "chunk_index": chunk.chunk_index,
                "offset": chunk.offset,
                "sha": chunk.sha,
                "text": chunk.text,
            }
            points.append(
                rest.PointStruct(
                    id=_make_point_id(
                        chunk,
                        embedding_model=embedding_model,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    ),
                    vector=vector,
                    payload=payload,
                )
            )
            if len(points) >= batch_size:
                self._upsert_batch(points, expected_dim)
                points = []
        if points:
            self._upsert_batch(points, expected_dim)

    def point_id_for_chunk(
        self,
        chunk: Chunk,
        *,
        embedding_model: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> str:
        return _make_point_id(
            chunk,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def _upsert_batch(self, points: List[rest.PointStruct], vector_dim: int) -> None:
        logger.info(
            "[UPSERT] collection=%s points=%d dim=%d",
            self.collection_name,
            len(points),
            vector_dim,
        )
        try:
            self._client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )
        except Exception:
            logger.exception("Failed to upsert batch with %d points", len(points))
            raise

    def count_points(self) -> int:
        try:
            response = self._client.count(
                collection_name=self.collection_name, exact=True
            )
        except Exception:
            logger.exception("Failed to count points for collection %s", self.collection_name)
            raise
        return int(getattr(response, "count", 0))

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

    def vector_size_for_model(self, model: str) -> int:
        return _vector_size_for_model(model)


def _make_point_id(
    chunk: Chunk,
    *,
    embedding_model: str,
    chunk_size: int,
    chunk_overlap: int,
) -> str:
    """Create a deterministic UUID point id for the provided chunk."""

    source = "|".join(
        [
            chunk.path,
            str(chunk.chunk_index),
            chunk.sha,
            embedding_model,
            str(chunk_size),
            str(chunk_overlap),
        ]
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, source))


def _vector_size_for_model(model: str) -> int:
    if model in _EMBEDDING_DIMENSIONS:
        return _EMBEDDING_DIMENSIONS[model]
    logger.warning(
        "Unknown embedding model %s; defaulting vector size to 1536", model
    )
    return 1536


def _mask_api_key(key: Optional[str]) -> str:
    if not key:
        return "<none>"
    if len(key) <= 4:
        return "***"
    return f"{key[:2]}...{key[-2:]}"
