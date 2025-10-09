from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from .legal_types import ChunkRecord
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
        vectors_config = _make_vector_config(vector_size)

        if recreate:
            logger.info(
                "Recreating collection %s with dimension %d", collection, vector_size
            )
            self._client.recreate_collection(
                collection_name=collection,
                vectors_config=vectors_config,
            )
            self._ensure_payload_indexes()
            return

        try:
            info = self._client.get_collection(collection_name=collection)
        except Exception:
            info = None

        if not info:
            logger.info(
                "Collection %s missing; creating with dimension %d", collection, vector_size
            )
            self._client.recreate_collection(
                collection_name=collection,
                vectors_config=vectors_config,
            )
            self._ensure_payload_indexes()
            return

        if not _collection_matches(info, vector_size):
            logger.warning(
                "Collection %s schema differs from expected; recreating", collection
            )
            self._client.recreate_collection(
                collection_name=collection,
                vectors_config=vectors_config,
            )
            self._ensure_payload_indexes()
        else:
            self._ensure_payload_indexes()

    def _ensure_payload_indexes(self) -> None:
        collection = self.collection_name
        keyword = _payload_schema("keyword")
        text = _payload_schema("text")
        index_specs = [
            ("doc_id", keyword),
            ("hierarchy.article_no", keyword),
            ("hierarchy.chapter_no", keyword),
            ("hierarchy.section_roman", keyword),
            ("law_meta.status", keyword),
            ("law_meta.last_amend_date", keyword),
            ("title_text", text),
            ("body_text", text),
        ]
        for field, schema in index_specs:
            try:
                self._client.create_payload_index(
                    collection_name=collection,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception as exc:
                logger.debug("Payload index %s not created: %s", field, exc)

    def delete_documents(self, doc_ids: Sequence[str]) -> None:
        doc_ids = [doc_id for doc_id in doc_ids if doc_id]
        if not doc_ids:
            return
        conditions = [
            rest.FieldCondition(key="doc_id", match=rest.MatchValue(value=doc_id))
            for doc_id in doc_ids
        ]
        flt = rest.Filter(should=conditions)
        try:
            selector = getattr(rest, "FilterSelector", None)
            if selector is None:
                raise AttributeError("FilterSelector is not available in qdrant_client models")
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=selector(filter=flt),
                wait=True,
            )
        except Exception:
            logger.exception("Failed to delete documents %s", doc_ids)
            raise

    def upsert_chunks(
        self,
        chunks: Sequence[ChunkRecord],
        title_vectors: Dict[str, List[float]],
        body_vectors: Dict[str, List[float]],
    ) -> None:
        if not chunks:
            return
        expected_dim = len(next(iter(body_vectors.values()))) if body_vectors else 0
        batch_size = max(1, self.settings.qdrant.upsert_batch_size)
        points: List[rest.PointStruct] = []
        for chunk in chunks:
            title_vector = title_vectors.get(chunk.title_sha256)
            body_vector = body_vectors.get(chunk.body_sha256)
            if title_vector is None or body_vector is None:
                raise ValueError(
                    f"Missing vectors for chunk {chunk.chunk_id} (title/body)"
                )
            if expected_dim and len(body_vector) != expected_dim:
                raise ValueError("Body vector dimension mismatch")
            payload = {
                "doc_id": chunk.doc_id,
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
                "title_text": chunk.title_text,
                "body_text": chunk.body_text,
                "hierarchy": chunk.hierarchy,
                "law_meta": chunk.law_meta,
                "source": chunk.source,
                "chunk_sha256": chunk.chunk_sha256,
                "title_sha256": chunk.title_sha256,
                "body_sha256": chunk.body_sha256,
            }
            point = rest.PointStruct(
                id=chunk.chunk_id,
                vector={
                    "title_vec": title_vector,
                    "body_vec": body_vector,
                },
                payload=payload,
            )
            points.append(point)
            if len(points) >= batch_size:
                self._upsert_batch(points, expected_dim)
                points = []
        if points:
            self._upsert_batch(points, expected_dim)

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

    def search(
        self,
        vector_name: str,
        query_vector: List[float],
        limit: int,
        *,
        filters: rest.Filter | None = None,
    ) -> List[rest.ScoredPoint]:
        return self._client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            vector_name=vector_name,
            limit=limit,
            with_payload=True,
            filter=filters,
        )

    def build_keyword_filter(self, query: str) -> rest.Filter:
        return rest.Filter(
            should=[
                rest.FieldCondition(key="title_text", match=rest.MatchText(text=query)),
                rest.FieldCondition(key="body_text", match=rest.MatchText(text=query)),
            ]
        )

    def test_connection(self) -> bool:
        try:
            status = self._client.get_collection(self.collection_name)
            return status is not None
        except Exception:
            return False

    def vector_size_for_model(self, model: str) -> int:
        return _vector_size_for_model(model)


def _make_vector_config(vector_size: int):
    try:
        return rest.VectorParamsMap(
            {
                "title_vec": rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE),
                "body_vec": rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE),
            }
        )
    except AttributeError:
        return {
            "title_vec": rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE),
            "body_vec": rest.VectorParams(size=vector_size, distance=rest.Distance.COSINE),
        }


def _collection_matches(info: object, expected_size: int) -> bool:
    params = getattr(getattr(info, "config", None), "params", None)
    vectors = getattr(params, "vectors", None)
    if not vectors:
        return False

    expected_names = {"title_vec", "body_vec"}
    actual_names: set[str] = set()

    try:
        items = list(vectors.items())  # type: ignore[attr-defined]
    except AttributeError:
        # Legacy single-vector collections expose a VectorParams instance
        # instead of a mapping. Force recreation to add named vectors.
        return False

    for name, vector in items:
        if name is None:
            return False
        actual_names.add(str(name))
        size = getattr(vector, "size", None)
        if size != expected_size:
            return False

    return actual_names == expected_names


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


def _payload_schema(name: str):
    schema_enum = getattr(rest, "PayloadSchemaType", None)
    if schema_enum is None:
        return name.lower()
    value = getattr(schema_enum, name.upper(), None)
    return value if value is not None else name.lower()

