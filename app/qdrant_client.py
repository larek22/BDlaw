from __future__ import annotations

import logging
import uuid
from typing import Dict, Iterator, List, Optional, Sequence, Set

import httpx
import importlib.metadata
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from .id_utils import make_point_id
from .legal_types import ChunkRecord
from datetime import datetime
from typing import Tuple

from .settings import AppSettings

logger = logging.getLogger(__name__)

try:
    qdrant_client_version = importlib.metadata.version("qdrant-client")
except importlib.metadata.PackageNotFoundError:
    qdrant_client_version = "unknown"

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
            "prefer_grpc": False,
        }

        self._endpoint_url = url
        self._alias_name = settings.qdrant.collection or "kb_docs_active"
        self._allow_destructive = getattr(
            settings, "allow_destructive_migrations", False
        )
        self._timeout = settings.qdrant.timeout_seconds

        self._api_key: Optional[str] = None
        if url.startswith("https://"):
            if not api_key:
                raise ValueError("Qdrant API key is required for HTTPS endpoints")
            client_kwargs["api_key"] = api_key
            self._api_key = api_key
        elif url.startswith("http://"):
            if "localhost" not in url and "127.0.0.1" not in url and api_key:
                client_kwargs["api_key"] = api_key
                self._api_key = api_key
        else:
            raise ValueError(f"Unsupported Qdrant URL: {url}")

        logger.info(
            "Using Qdrant endpoint: %s (api_key=%s)",
            url,
            _mask_api_key(client_kwargs.get("api_key")),
        )

        self._client = QdrantClient(**client_kwargs)
        self._server_version = self._fetch_server_version()
        logger.info(
            "Qdrant runtime: client=%s server=%s embedding_model=%s",
            qdrant_client_version,
            self._server_version or "<unknown>",
            settings.openai_models.embedding,
        )

    @property
    def collection_name(self) -> str:
        return self._alias_name

    @property
    def endpoint_url(self) -> str:
        return self._endpoint_url

    @property
    def server_version(self) -> Optional[str]:
        return self._server_version

    @property
    def client(self) -> QdrantClient:
        return self._client

    def ensure_collection(self, embedding_model: str, recreate: bool = False) -> None:
        alias = self.collection_name
        vector_size = _vector_size_for_model(embedding_model)

        if recreate:
            logger.info(
                "Provisioning fresh collection for alias %s with dim=%d", alias, vector_size
            )
            previous = self._resolve_alias(alias)
            new_collection = self._provision_collection(alias, vector_size)
            self._swap_alias(alias, new_collection, previous)
            if (
                previous
                and previous not in {alias, new_collection}
                and self._allow_destructive
            ):
                self._delete_collection(previous)
            self._ensure_payload_indexes(new_collection)
            return

        current, info, alias_exists = self._current_collection_info(alias)
        if info and _collection_matches(info, vector_size):
            logger.debug("Collection %s already matches expected schema", current)
            self._ensure_payload_indexes(current)
            return

        logger.info(
            "Creating collection for alias %s with vector dimension %d", alias, vector_size
        )
        new_collection = self._provision_collection(alias, vector_size)
        previous = current if alias_exists else self._resolve_alias(alias)
        self._swap_alias(alias, new_collection, previous)
        if (
            previous
            and previous not in {alias, new_collection}
            and self._allow_destructive
        ):
            self._delete_collection(previous)
        self._ensure_payload_indexes(new_collection)

    def reset_collection(self, embedding_model: str) -> None:
        if not self._allow_destructive:
            logger.info(
                "Reset requested with destructive migrations disabled; provisioning fresh collection via alias swap."
            )
        else:
            logger.info("Resetting active collection via alias rotation.")
        self.ensure_collection(embedding_model=embedding_model, recreate=True)

    def _ensure_payload_indexes(self, collection: str | None = None) -> None:
        collection = collection or self.collection_name
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
                message = str(exc).lower()
                if "exists" in message or "already" in message:
                    logger.debug("Payload index %s already present", field)
                else:
                    logger.warning("Payload index %s not created: %s", field, exc)
            else:
                logger.info("Payload index ensured for field '%s'", field)

    def keyword_prefilter(self, query: str, limit: int = 0) -> List[str]:
        if limit <= 0:
            return []
        filter_ = self.build_keyword_filter(query)
        seen: set[str] = set()
        doc_ids: List[str] = []
        offset = None
        remaining = limit
        while remaining > 0:
            batch_limit = min(128, remaining)
            try:
                points, offset = self._client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=filter_,
                    limit=batch_limit,
                    offset=offset,
                    with_payload=True,
                )
            except Exception as exc:
                logger.warning("Keyword prefilter failed: %s", exc)
                return []
            if not points:
                break
            for point in points:
                payload = point.payload or {}
                doc_id = payload.get("doc_id")
                if doc_id and doc_id not in seen:
                    seen.add(str(doc_id))
                    doc_ids.append(str(doc_id))
                    remaining -= 1
                    if remaining <= 0:
                        break
            if not offset:
                break
        return doc_ids

    def build_doc_id_filter(self, doc_ids: Sequence[str]) -> rest.Filter:
        cleaned = [doc_id for doc_id in doc_ids if doc_id]
        if not cleaned:
            raise ValueError("doc_ids must not be empty")
        match_any = getattr(rest, "MatchAny", None)
        if match_any is not None:
            return rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="doc_id",
                        match=match_any(any=cleaned),
                    )
                ]
            )
        return rest.Filter(
            should=[
                rest.FieldCondition(key="doc_id", match=rest.MatchValue(value=value))
                for value in cleaned
            ]
        )

    def build_as_of_filter(
        self,
        *,
        status: Optional[str] = None,
        as_of_start: Optional[str] = None,
        as_of_date: Optional[str] = None,
    ) -> Optional[rest.Filter]:
        conditions: List[rest.FieldCondition] = []
        if status:
            conditions.append(
                rest.FieldCondition(
                    key="law_meta.status",
                    match=rest.MatchValue(value=status),
                )
            )
        if as_of_start or as_of_date:
            range_kwargs: Dict[str, str] = {}
            if as_of_start:
                range_kwargs["gte"] = as_of_start
            if as_of_date:
                range_kwargs["lte"] = as_of_date
            conditions.append(
                rest.FieldCondition(
                    key="law_meta.last_amend_date",
                    range=rest.Range(**range_kwargs),
                )
            )
        if not conditions:
            return None
        return rest.Filter(must=conditions)

    def combine_filters(self, *filters: Optional[rest.Filter]) -> Optional[rest.Filter]:
        active = [flt for flt in filters if flt is not None]
        if not active:
            return None
        if len(active) == 1:
            return active[0]

        must_conditions: List = []
        should_conditions: List = []
        must_not_conditions: List = []

        for flt in active:
            must_conditions.extend(list(getattr(flt, "must", []) or []))
            should_conditions.extend(list(getattr(flt, "should", []) or []))
            must_not_conditions.extend(list(getattr(flt, "must_not", []) or []))

        return rest.Filter(
            must=must_conditions or None,
            should=should_conditions or None,
            must_not=must_not_conditions or None,
        )

    def delete_documents(self, doc_ids: Sequence[str]) -> None:
        doc_ids = [doc_id for doc_id in doc_ids if doc_id]
        if not doc_ids:
            return
        flt = self.build_doc_id_filter(doc_ids)
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

    def count_points_for_doc_ids(self, doc_ids: Sequence[str]) -> Dict[str, int]:
        cleaned = [doc_id for doc_id in doc_ids if doc_id]
        if not cleaned:
            return {}
        try:
            flt = self.build_doc_id_filter(cleaned)
        except ValueError:
            return {}

        counts: Dict[str, int] = {doc_id: 0 for doc_id in cleaned}
        offset = None

        while True:
            try:
                points, offset = self._client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=flt,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                )
            except Exception:
                logger.exception("Failed to scroll for document counts in %s", self.collection_name)
                raise
            if not points:
                break
            for point in points:
                payload = getattr(point, "payload", {}) or {}
                doc_id = str(payload.get("doc_id") or "")
                if doc_id in counts:
                    counts[doc_id] += 1
            if not offset:
                break

        return counts

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
            point_id = chunk.chunk_id or make_point_id(chunk.doc_id, chunk.chunk_index)
            try:
                uuid.UUID(point_id)
            except ValueError as exc:  # pragma: no cover - defensive guard
                raise ValueError(
                    f"Generated invalid UUID for chunk {chunk.doc_id}:{chunk.chunk_index}"
                ) from exc
            if chunk.chunk_id != point_id:
                chunk.chunk_id = point_id
            payload = {
                "doc_id": chunk.doc_id,
                "chunk_id": point_id,
                "chunk_index": chunk.chunk_index,
                 "chunk_key": chunk.chunk_key,
                "title_text": chunk.title_text,
                "body_text": chunk.body_text,
                "hierarchy": chunk.hierarchy,
                "law_meta": chunk.law_meta,
                "source": chunk.source,
                "plan_version": chunk.plan_version,
                "parser_version": chunk.parser_version,
                "chunk_sha256": chunk.chunk_sha256,
                "title_sha256": chunk.title_sha256,
                "body_sha256": chunk.body_sha256,
            }
            point = rest.PointStruct(
                id=point_id,
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

    def _current_collection_info(self, alias: str) -> Tuple[Optional[str], Optional[object], bool]:
        target = self._resolve_alias(alias)
        alias_exists = target is not None
        lookup = target or alias
        try:
            info = self._client.get_collection(collection_name=lookup)
        except Exception:
            info = None
        if info:
            return lookup, info, alias_exists
        return target, None, alias_exists

    def _provision_collection(self, alias: str, vector_size: int) -> str:
        base = f"{alias}_d{vector_size}_{datetime.utcnow():%Y%m%d_%H%M%S}"
        candidate = base
        suffix = 0
        vectors_config = _make_vector_config(vector_size)
        while True:
            try:
                self._client.create_collection(
                    collection_name=candidate,
                    vectors_config=vectors_config,
                    hnsw_config=_make_hnsw_config(),
                    optimizers_config=_make_optimizer_config(),
                )
            except Exception as exc:
                message = str(exc).lower()
                if "exists" in message or "already" in message:
                    suffix += 1
                    candidate = f"{base}_{suffix}"
                    continue
                logger.exception("Failed to create collection %s", candidate)
                raise
            logger.info("Created collection %s for alias %s", candidate, alias)
            return candidate

    def _resolve_alias(self, alias: str) -> Optional[str]:
        base_url = _normalise_base_url(self._endpoint_url)
        url = f"{base_url}/aliases/{alias}"
        headers = _build_headers(self._api_key)
        try:
            response = httpx.get(url, headers=headers, timeout=self._timeout)
        except Exception as exc:
            logger.debug("Failed to resolve alias %s: %s", alias, exc)
            return None
        if response.status_code != 200:
            return None
        try:
            payload = response.json()
        except ValueError:
            return None
        if isinstance(payload, dict):
            result = payload.get("result")
            if isinstance(result, dict):
                collection = result.get("collection_name") or result.get("collection")
                if isinstance(collection, str):
                    return collection
        return None

    def _swap_alias(self, alias: str, new_collection: str, previous: Optional[str]) -> None:
        base_url = _normalise_base_url(self._endpoint_url)
        url = f"{base_url}/aliases"
        headers = _build_headers(self._api_key)
        actions = [
            {"create_alias": {"alias_name": alias, "collection_name": new_collection}}
        ]
        if previous and previous != new_collection:
            actions.append({"delete_alias": {"alias_name": alias}})
        payload = {"actions": actions}
        try:
            response = httpx.post(url, headers=headers, json=payload, timeout=self._timeout)
            response.raise_for_status()
        except Exception:
            logger.exception(
                "Failed to update alias %s to point at %s", alias, new_collection
            )
            raise
        logger.info("Alias %s now points to collection %s", alias, new_collection)

    def _delete_collection(self, collection: str) -> None:
        try:
            self._client.delete_collection(collection_name=collection)
        except Exception as exc:
            logger.warning("Failed to delete legacy collection %s: %s", collection, exc)


    def count_points(self) -> int:
        try:
            response = self._client.count(
                collection_name=self.collection_name, exact=True
            )
        except Exception:
            logger.exception("Failed to count points for collection %s", self.collection_name)
            raise
        return int(getattr(response, "count", 0))

    def count_points_with_prefix(self, prefix: str) -> int:
        if not prefix:
            return 0
        count = 0
        offset = None
        while True:
            try:
                points, offset = self._client.scroll(
                    collection_name=self.collection_name,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                )
            except Exception:
                logger.exception(
                    "Failed to scroll points for prefix reconciliation in %s",
                    self.collection_name,
                )
                raise
            if not points:
                break
            for point in points:
                payload = getattr(point, "payload", {}) or {}
                doc_id = str(payload.get("doc_id") or "")
                if doc_id.startswith(prefix):
                    count += 1
            if not offset:
                break
        return count

    def purge_orphans(self, prefixes: Sequence[str]) -> int:
        allowed = [prefix for prefix in prefixes if prefix]
        removed = 0
        offset = None
        batch: List[str] = []

        def flush() -> None:
            nonlocal removed, batch
            valid_ids = [point_id for point_id in batch if point_id]
            if not valid_ids:
                batch = []
                return
            try:
                self._client.delete(
                    collection_name=self.collection_name,
                    points_selector=valid_ids,
                    wait=True,
                )
            except Exception:
                logger.exception("Failed to delete orphaned points: %s", valid_ids)
                raise
            removed += len(valid_ids)
            batch = []

        while True:
            try:
                points, offset = self._client.scroll(
                    collection_name=self.collection_name,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                )
            except Exception:
                logger.exception(
                    "Failed to scroll for orphan purge in %s", self.collection_name
                )
                raise
            if not points:
                break
            for point in points:
                payload = getattr(point, "payload", {}) or {}
                doc_id = str(payload.get("doc_id") or "")
                keep = bool(doc_id)
                if keep and allowed:
                    keep = any(doc_id.startswith(prefix) for prefix in allowed)
                if not keep:
                    batch.append(str(getattr(point, "id", "")))
                if len(batch) >= 256:
                    flush()
            if not offset:
                break
        if batch:
            flush()
        return removed

    def query(
        self,
        vector_name: str,
        query_vector: Sequence[float],
        limit: int,
        *,
        filters: rest.Filter | None = None,
    ) -> List[rest.ScoredPoint]:
        response = self._client.query_points(
            collection_name=self.collection_name,
            query=list(query_vector),
            query_filter=filters,
            using=vector_name,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        points = getattr(response, "points", response)
        return list(points)

    def fetch_article_chunks(self, article_no: str, *, limit: int = 3) -> List[ChunkRecord]:
        if not article_no:
            return []
        filter_ = rest.Filter(
            must=[
                rest.FieldCondition(
                    key="hierarchy.article_no",
                    match=rest.MatchValue(value=article_no),
                )
            ]
        )
        try:
            points, _ = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=filter_,
                limit=max(1, limit * 3),
                with_payload=True,
            )
        except Exception:
            logger.exception("Failed to fetch article chunks for %s", article_no)
            raise
        chunks = [
            self._payload_to_chunk(getattr(point, "payload", {}) or {})
            for point in points
            if getattr(point, "payload", None)
        ]
        chunks.sort(key=lambda chunk: chunk.chunk_index)
        if limit > 0:
            chunks = chunks[:limit]
        return chunks

    def fetch_neighbor_chunks(
        self, doc_id: str, chunk_index: int, window: int = 1
    ) -> List[ChunkRecord]:
        if window <= 0 or not doc_id:
            return []
        try:
            flt = rest.Filter(
                must=[
                    rest.FieldCondition(
                        key="doc_id", match=rest.MatchValue(value=doc_id)
                    ),
                    rest.FieldCondition(
                        key="chunk_index",
                        range=rest.Range(
                            gte=max(0, chunk_index - window),
                            lte=chunk_index + window,
                        ),
                    ),
                ]
            )
            points, _ = self._client.scroll(
                collection_name=self.collection_name,
                scroll_filter=flt,
                limit=window * 2 + 1,
                with_payload=True,
            )
        except Exception:
            logger.exception("Failed to fetch neighbor chunks for %s", doc_id)
            raise
        neighbors: List[ChunkRecord] = []
        for point in points:
            payload = getattr(point, "payload", {}) or {}
            record = self._payload_to_chunk(payload)
            if record.chunk_index != chunk_index:
                neighbors.append(record)
        neighbors.sort(key=lambda chunk: chunk.chunk_index)
        return neighbors

    def build_keyword_filter(self, query: str) -> rest.Filter:
        return rest.Filter(
            should=[
                rest.FieldCondition(key="title_text", match=rest.MatchText(text=query)),
                rest.FieldCondition(key="body_text", match=rest.MatchText(text=query)),
            ]
        )

    @staticmethod
    def _payload_to_chunk(payload: Dict[str, object]) -> ChunkRecord:
        def _ensure_dict(value: object) -> Dict[str, object]:
            if isinstance(value, dict):
                return dict(value)
            if hasattr(value, "items"):
                try:
                    return dict(value.items())  # type: ignore[attr-defined]
                except Exception:
                    return {}
            return {}

        return ChunkRecord(
            doc_id=str(payload.get("doc_id", "")),
            chunk_index=int(payload.get("chunk_index", 0)),
            chunk_id=str(payload.get("chunk_id", "")),
            title_text=str(payload.get("title_text", "")),
            body_text=str(payload.get("body_text", "")),
            hierarchy=_ensure_dict(payload.get("hierarchy")),
            law_meta=_ensure_dict(payload.get("law_meta")),
            source=_ensure_dict(payload.get("source")),
            chunk_sha256=str(payload.get("chunk_sha256", "")),
            title_sha256=str(payload.get("title_sha256", "")),
            body_sha256=str(payload.get("body_sha256", "")),
        )

    def iter_doc_ids(
        self,
        *,
        batch_size: int = 256,
        limit: Optional[int] = None,
    ) -> Iterator[str]:
        """Yield unique doc_id values stored in the collection."""

        seen: Set[str] = set()
        fetched = 0
        offset = None

        while True:
            if limit is not None and fetched >= limit:
                break
            request_limit = batch_size
            if limit is not None:
                request_limit = min(request_limit, max(0, limit - fetched))
                if request_limit == 0:
                    break
            try:
                points, offset = self._client.scroll(
                    collection_name=self.collection_name,
                    limit=request_limit,
                    offset=offset,
                    with_payload=True,
                )
            except Exception:
                logger.exception(
                    "Failed to scroll doc ids from collection %s", self.collection_name
                )
                raise

            if not points:
                break

            for point in points:
                payload = getattr(point, "payload", {}) or {}
                doc_id = payload.get("doc_id")
                if not doc_id:
                    continue
                doc_id_str = str(doc_id)
                if doc_id_str in seen:
                    continue
                seen.add(doc_id_str)
                fetched += 1
                yield doc_id_str

            if not offset:
                break

    def _fetch_server_version(self) -> Optional[str]:
        base = _normalise_base_url(self._endpoint_url)
        for path in ("/version", "/telemetry", "/readyz", "/healthz"):
            candidate = _safe_request(f"{base}{path}", self._api_key)
            if candidate:
                return candidate
        return None

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


def _make_hnsw_config() -> rest.HnswConfigDiff:
    return rest.HnswConfigDiff(m=64, ef_construct=512)


def _make_optimizer_config() -> rest.OptimizersConfigDiff:
    return rest.OptimizersConfigDiff(memmap_threshold=20000)


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


def _extract_version_from_json(data: object) -> Optional[str]:
    if isinstance(data, dict):
        if "version" in data and isinstance(data["version"], str):
            return data["version"]
        telemetry = data.get("result")
        if isinstance(telemetry, dict):
            version = telemetry.get("app", {}).get("version")
            if isinstance(version, str):
                return version
        app = data.get("app")
        if isinstance(app, dict):
            version = app.get("version")
            if isinstance(version, str):
                return version
    return None


def _fetch_json(response: httpx.Response) -> Optional[str]:
    try:
        data = response.json()
    except ValueError:
        text = response.text.strip()
        return text[:120] if text else None
    return _extract_version_from_json(data)


def _normalise_base_url(url: str) -> str:
    return url.rstrip("/")


def _build_headers(api_key: Optional[str]) -> Dict[str, str]:
    if not api_key:
        return {}
    return {"api-key": api_key}


def _safe_request(url: str, api_key: Optional[str]) -> Optional[str]:
    headers = _build_headers(api_key)
    try:
        response = httpx.get(url, headers=headers, timeout=5.0)
        response.raise_for_status()
    except Exception as exc:
        logger.debug("Failed to query %s: %s", url, exc)
        return None
    return _fetch_json(response)


