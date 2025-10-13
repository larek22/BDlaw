from __future__ import annotations

import logging
import random
import time
import uuid
from datetime import datetime
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple
from urllib.parse import urlparse

import httpx
import importlib.metadata
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
try:  # pragma: no cover - optional import varies by qdrant-client version
    from qdrant_client.http.exceptions import ResponseHandlingException as _ResponseHandlingException
except Exception:  # pragma: no cover - older clients
    _ResponseHandlingException = ()

from .id_utils import make_point_id
from .legal_types import ChunkRecord
from .settings import AppSettings
from .utils.net import normalize_qdrant_url

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
        raw_url = (settings.qdrant.url or "").strip()
        url = normalize_qdrant_url(raw_url)
        api_key = (settings.qdrant.api_key or "").strip()

        if not url:
            raise ValueError("Qdrant URL must be configured")

        client_kwargs: dict[str, object] = {
            "url": url,
            "timeout": settings.qdrant.timeout_seconds,
            "prefer_grpc": settings.qdrant.prefer_grpc,
        }
        httpx_client: httpx.Client | None = None
        try:
            timeout = httpx.Timeout(
                timeout=settings.qdrant.read_timeout_seconds,
                connect=settings.qdrant.connect_timeout_seconds,
                read=settings.qdrant.read_timeout_seconds,
                write=settings.qdrant.write_timeout_seconds,
            )
            limits = httpx.Limits(max_connections=20, max_keepalive_connections=10)
            httpx_client = httpx.Client(
                timeout=timeout,
                limits=limits,
                http2=True,
            )
            client_kwargs["httpx_client"] = httpx_client
        except AttributeError:
            logger.warning("httpx.Timeout unavailable; using scalar timeout configuration")
        self._httpx_client = httpx_client
        if settings.qdrant.prefer_grpc:
            client_kwargs["grpc_port"] = settings.qdrant.grpc_port

        self._endpoint_url = url
        self._alias_name = settings.qdrant.collection or "kb_docs_active"
        self._write_collection_name = self._alias_name
        self._allow_destructive = getattr(
            settings, "allow_destructive_migrations", False
        )
        self._timeout = settings.qdrant.timeout_seconds

        self._api_key: Optional[str] = None
        if url.startswith("https://"):
            parsed_url = urlparse(url)
            host = (parsed_url.hostname or "").lower()
            if not api_key:
                if host in {"localhost", "127.0.0.1", "::1"} or settings.qdrant.allow_insecure_https_without_api_key:
                    logger.warning(
                        "HTTPS endpoint %s missing API key; proceeding for local testing", url
                    )
                else:
                    raise ValueError("Qdrant API key is required for HTTPS endpoints")
            else:
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

        try:
            self._client = QdrantClient(**client_kwargs)
        except TypeError as exc:
            if "httpx_client" in client_kwargs and "httpx_client" in str(exc):
                logger.warning(
                    "Qdrant client rejected custom httpx_client; retrying with scalar timeouts"
                )
                httpx_obj = client_kwargs.pop("httpx_client", None)
                # ensure the bespoke client does not leak resources if unused
                if httpx_obj is not None:
                    try:
                        httpx_obj.close()
                    except Exception:  # pragma: no cover - best effort cleanup
                        logger.debug("Failed to close unused httpx client", exc_info=True)
                self._httpx_client = None
                self._client = QdrantClient(**client_kwargs)
            else:
                raise
        self._server_version = self._fetch_server_version()
        logger.info(
            "Qdrant runtime: client=%s server=%s embedding_model=%s",
            qdrant_client_version,
            self._server_version or "<unknown>",
            settings.openai_models.embedding,
        )
        self._target_batch_bytes = max(1024, settings.qdrant.target_batch_bytes)
        self._title_max_chars = max(0, settings.qdrant.title_max_chars)
        self._body_max_chars = max(0, settings.qdrant.preview_max_chars)
        self._max_retry_attempts = max(1, settings.qdrant.max_retries)
        self._retry_initial_delay = max(0.1, settings.qdrant.retry_initial_delay)
        self._retry_max_delay = max(
            self._retry_initial_delay, settings.qdrant.retry_max_delay
        )
        self._max_batch_size = max(1, settings.qdrant.max_batch_size)
        self._min_batch_size = max(1, settings.qdrant.min_batch_size)
        if self._min_batch_size > self._max_batch_size:
            self._min_batch_size = self._max_batch_size
        self._wait_for_upsert = settings.qdrant.use_wait
        self._ordering_preference = (settings.qdrant.upsert_ordering or "weak").strip().lower()
        self._prefer_grpc = bool(
            getattr(getattr(self._client, "_client", None), "_prefer_grpc", settings.qdrant.prefer_grpc)
        )
        self._ordering = self._resolve_write_ordering(self._ordering_preference)
        self._validation_max_wait_seconds = max(
            0.0, settings.ingest.validation_max_wait_seconds
        )
        self._validation_poll_interval_seconds = max(
            0.1, settings.ingest.validation_poll_interval_seconds
        )

    @property
    def collection_name(self) -> str:
        return self._write_collection_name

    @property
    def alias_name(self) -> str:
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

    def _resolve_write_ordering(self, preference: str | None) -> object | None:
        if not preference:
            return None
        normalized = preference.strip().lower()
        aliases = {
            "weak": ("Weak", "WEAK"),
            "medium": ("Medium", "MEDIUM"),
            "strong": ("Strong", "STRONG"),
        }
        if normalized not in aliases:
            logger.warning(
                "Unknown write ordering '%s'; defaulting to 'weak' for compatibility",
                preference,
            )
            normalized = "weak"
        grpc_name, rest_name = aliases[normalized]
        if self._prefer_grpc:
            try:
                from qdrant_client.grpc import points_pb2

                value = points_pb2.WriteOrderingType.Value(grpc_name)
                return points_pb2.WriteOrdering(type=value)
            except Exception:  # pragma: no cover - dependent on installed qdrant-client
                logger.warning(
                    "gRPC write ordering '%s' unavailable; falling back to default", grpc_name,
                )
                return None
        try:
            from qdrant_client.http import models as rest

            return getattr(rest.WriteOrdering, rest_name)
        except Exception:  # pragma: no cover - dependent on installed qdrant-client
            logger.debug(
                "REST write ordering '%s' unavailable; using plain preference string",
                rest_name,
                exc_info=True,
            )
            return normalized

    def ensure_collection(self, embedding_model: str, recreate: bool = False) -> None:
        alias = self.alias_name
        vector_size = _vector_size_for_model(embedding_model)

        if recreate:
            logger.info(
                "Provisioning fresh collection for alias %s with dim=%d", alias, vector_size
            )
            previous = self._resolve_alias(alias)
            new_collection = self._provision_collection(
                alias, vector_size, on_disk=True
            )
            swapped = self._swap_alias(alias, new_collection, previous)
            if (
                previous
                and previous not in {alias, new_collection}
                and self._allow_destructive
            ):
                self._delete_collection(previous)
            target = alias if swapped else new_collection
            self._ensure_payload_indexes(target)
            return

        current, info, alias_exists = self._current_collection_info(alias)
        if alias_exists:
            self._write_collection_name = self._alias_name
        if info and _collection_matches(info, vector_size):
            logger.debug("Collection %s already matches expected schema", current)
            self._ensure_payload_indexes(current)
            return

        logger.info(
            "Creating collection for alias %s with vector dimension %d", alias, vector_size
        )
        new_collection = self._provision_collection(alias, vector_size, on_disk=True)
        previous = current if alias_exists else self._resolve_alias(alias)
        swapped = self._swap_alias(alias, new_collection, previous)
        if (
            previous
            and previous not in {alias, new_collection}
            and self._allow_destructive
        ):
            self._delete_collection(previous)
        target = alias if swapped else new_collection
        self._ensure_payload_indexes(target)

    def ensure_shadow_collection(self, alias: str, dim: int, on_disk: bool = True) -> str:
        logger.info(
            "Provisioning shadow collection for alias=%s dim=%d on_disk=%s",
            alias,
            dim,
            on_disk,
        )
        target = self._provision_collection(alias, dim, on_disk=on_disk)
        self._ensure_payload_indexes(target)
        logger.info("Shadow collection ready: %s", target)
        return target

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
        integer = _payload_schema("integer")
        index_specs = [
            ("doc_id", keyword),
            ("hierarchy.part_no", integer),
            ("hierarchy.chapter_no", integer),
            ("hierarchy.article_no_int", integer),
            ("hierarchy.section_roman", keyword),
            ("law_meta.status", keyword),
            ("law_meta.enact_date_int", integer),
            ("law_meta.last_amend_date_int", integer),
            ("law_meta.date_from_int", integer),
            ("law_meta.date_to_int", integer),
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

    def payload_to_chunk(self, payload: Mapping[str, object]) -> ChunkRecord:
        raw_title = payload.get("title_text")
        title_text = raw_title if isinstance(raw_title, str) else ""
        raw_body = payload.get("body_text")
        body_text = raw_body if isinstance(raw_body, str) else ""

        if not title_text and body_text:
            title_text = body_text[:120].strip()
        if not body_text and title_text:
            body_text = title_text

        hierarchy = dict(payload.get("hierarchy") or {})
        law_meta = dict(payload.get("law_meta") or {})
        source = dict(payload.get("source") or {})

        return ChunkRecord(
            doc_id=str(payload.get("doc_id") or ""),
            chunk_index=int(payload.get("chunk_index") or 0),
            chunk_id=str(payload.get("chunk_id") or ""),
            title_text=title_text,
            body_text=body_text,
            hierarchy=hierarchy,
            law_meta=law_meta,
            source=source,
            chunk_key=payload.get("chunk_key"),
            plan_version=payload.get("plan_version"),
            parser_version=payload.get("parser_version"),
            chunk_sha256=str(payload.get("chunk_sha256") or ""),
            title_sha256=str(payload.get("title_sha256") or ""),
            body_sha256=str(payload.get("body_sha256") or ""),
        )

    def list_chapters(self) -> List[int]:
        seen: Set[int] = set()
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
                logger.exception("Failed to enumerate chapters from %s", self.collection_name)
                break
            if not points:
                break
            for point in points:
                payload = getattr(point, "payload", {}) or {}
                hierarchy = payload.get("hierarchy") or {}
                chapter = hierarchy.get("chapter_no") if isinstance(hierarchy, dict) else None
                if isinstance(chapter, int):
                    seen.add(chapter)
            if not offset:
                break
        return sorted(seen)

    def fetch_article_chunks(self, doc_id: str) -> List[ChunkRecord]:
        if not doc_id:
            return []
        try:
            flt = self.build_doc_id_filter([doc_id])
        except ValueError:
            return []

        records: List[ChunkRecord] = []
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
                logger.exception("Failed to load article chunks for %s", doc_id)
                break
            if not points:
                break
            for point in points:
                payload = getattr(point, "payload", {}) or {}
                records.append(self.payload_to_chunk(payload))
            if not offset:
                break

        records.sort(key=lambda chunk: chunk.chunk_index)
        return records

    def active_collection_target(self) -> str:
        target = self._resolve_alias(self._alias_name)
        return target or self.collection_name

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
            range_kwargs: Dict[str, int] = {}
            start_int = _date_string_to_int(as_of_start)
            end_int = _date_string_to_int(as_of_date)
            if start_int is not None:
                range_kwargs["gte"] = start_int
            if end_int is not None:
                range_kwargs["lte"] = end_int
            if range_kwargs:
                conditions.append(
                    rest.FieldCondition(
                        key="law_meta.last_amend_date_int",
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
        *,
        collection_name: str | None = None,
    ) -> None:
        if not chunks:
            return
        target_collection = collection_name or self.collection_name
        expected_dim = len(next(iter(body_vectors.values()))) if body_vectors else 0
        points: List[rest.PointStruct] = []
        estimated_bytes = 0
        target_bytes = self._target_batch_bytes
        logger.info(
            "Preparing dynamic upsert for %d chunk(s) into %s (expected_dim=%d)",
            len(chunks),
            target_collection,
            expected_dim,
        )
        for chunk in chunks:
            title_vector = title_vectors.get(chunk.title_sha256)
            body_vector = body_vectors.get(chunk.body_sha256)
            if title_vector is None or body_vector is None:
                raise ValueError(
                    f"Missing vectors for chunk {chunk.chunk_id} (title/body)"
                )
            if expected_dim and len(body_vector) != expected_dim:
                raise ValueError("Body vector dimension mismatch")
            article_no: str | None = None
            if isinstance(chunk.hierarchy, dict):
                raw_article = chunk.hierarchy.get("article_no")
                if raw_article is not None:
                    article_no = str(raw_article)
            point_id = chunk.chunk_id or make_point_id(
                chunk.doc_id,
                chunk.chunk_index,
                article_no=article_no,
            )
            try:
                uuid.UUID(point_id)
            except ValueError as exc:  # pragma: no cover - defensive guard
                raise ValueError(
                    f"Generated invalid UUID for chunk {chunk.doc_id}:{chunk.chunk_index}"
                ) from exc
            if chunk.chunk_id != point_id:
                chunk.chunk_id = point_id
            title_text = _truncate_text(chunk.title_text, self._title_max_chars)
            body_text = _truncate_text(chunk.body_text, self._body_max_chars)
            payload = {
                "doc_id": chunk.doc_id,
                "chunk_id": point_id,
                "chunk_index": chunk.chunk_index,
                "chunk_key": chunk.chunk_key,
                "title_text": title_text,
                "body_text": body_text,
                "hierarchy": chunk.hierarchy,
                "law_meta": chunk.law_meta,
                "source": chunk.source,
                "plan_version": chunk.plan_version,
                "parser_version": chunk.parser_version,
                "chunk_sha256": chunk.chunk_sha256,
                "title_sha256": chunk.title_sha256,
                "body_sha256": chunk.body_sha256,
            }
            payload = _slim_payload_if_needed(
                payload,
                enable_slim=self.settings.payload.slim_body_text,
            )
            point = rest.PointStruct(
                id=point_id,
                vector={
                    "title_vec": title_vector,
                    "body_vec": body_vector,
                },
                payload=payload,
            )
            point_bytes = self._estimate_point_bytes(title_vector, body_vector, payload)
            if points and estimated_bytes + point_bytes > target_bytes:
                self._upsert_points_dynamic(target_collection, points)
                points = []
                estimated_bytes = 0
            points.append(point)
            estimated_bytes += point_bytes
        if points:
            self._upsert_points_dynamic(target_collection, points)

    def _upsert_points_dynamic(
        self, collection: str, points: Sequence[rest.PointStruct]
    ) -> None:
        if not points:
            return
        total = len(points)
        batch_size = min(self._max_batch_size, max(1, total))
        index = 0
        attempt = 0
        retryable_types: List[type[BaseException]] = []
        write_timeout = getattr(httpx, "WriteTimeout", None)
        if isinstance(write_timeout, type):
            retryable_types.append(write_timeout)
        if isinstance(_ResponseHandlingException, type):
            retryable_types.append(_ResponseHandlingException)
        elif isinstance(_ResponseHandlingException, tuple):
            retryable_types.extend(
                exc for exc in _ResponseHandlingException if isinstance(exc, type)
            )
        if not retryable_types:
            retryable_types.append(Exception)
        retryable = tuple(retryable_types)

        while index < total:
            end = min(total, index + batch_size)
            batch = list(points[index:end])
            first_vector = getattr(batch[0], "vector", {}) if batch else {}
            body_vec = None
            if isinstance(first_vector, Mapping):
                body_vec = first_vector.get("body_vec")
            vector_dim = len(body_vec or []) if isinstance(body_vec, Sequence) else 0
            estimated_bytes = self._estimate_points_bytes(batch, vector_dim)
            logger.info(
                "Upsert batch=%d target=%s start=%d/%d", len(batch), collection, index, total
            )
            try:
                self._commit_upsert(
                    batch,
                    estimated_bytes,
                    collection_name=collection,
                )
            except retryable as exc:  # type: ignore[misc]
                if batch_size > self._min_batch_size:
                    batch_size = max(self._min_batch_size, batch_size // 2)
                    attempt += 1
                    backoff = min(
                        self._retry_max_delay,
                        self._retry_initial_delay * (2**attempt),
                    )
                    sleep_for = backoff + random.random()
                    logger.warning(
                        "Upsert batch=%d failed for %s; shrinking batch to %d and retrying in %.2fs (%s)",
                        len(batch),
                        collection,
                        batch_size,
                        sleep_for,
                        exc,
                    )
                    time.sleep(sleep_for)
                    continue
                logger.exception(
                    "Dynamic upsert exhausted retries for %s at batch size %d", collection, batch_size
                )
                raise
            else:
                logger.info(
                    "Upsert batch=%d succeeded for %s; shrinking_attempt=%d",
                    len(batch),
                    collection,
                    attempt,
                )
                index = end
                attempt = 0

    def _upsert_batch(
        self,
        points: List[rest.PointStruct],
        vector_dim: int,
        estimated_bytes: float,
        *,
        collection_name: str | None = None,
    ) -> None:
        # DEPRECATED: legacy recursive splitter preserved for compatibility.
        logger.info(
            "[UPSERT] collection=%s points=%d dim=%d bytes~=%.0f",
            collection_name or self.collection_name,
            len(points),
            vector_dim,
            estimated_bytes,
        )
        try:
            self._commit_upsert(
                points,
                estimated_bytes,
                collection_name=collection_name,
            )
        except Exception as exc:
            if len(points) > 1 and _is_timeout_error(exc):
                split = max(1, len(points) // 2)
                logger.warning(
                    "Upsert timed out for %d points; retrying as batches of %d and %d",
                    len(points),
                    split,
                    len(points) - split,
                )
                left = points[:split]
                right = points[split:]
                self._upsert_batch(
                    left,
                    vector_dim,
                    self._estimate_points_bytes(left, vector_dim),
                    collection_name=collection_name,
                )
                self._upsert_batch(
                    right,
                    vector_dim,
                    self._estimate_points_bytes(right, vector_dim),
                    collection_name=collection_name,
                )
                return
            logger.exception("Failed to upsert batch with %d points", len(points))
            raise

    def _commit_upsert(
        self,
        points: List[rest.PointStruct],
        estimated_bytes: float,
        *,
        collection_name: str | None = None,
    ) -> None:
        target_collection = collection_name or self.collection_name
        attempt = 0
        delay = self._retry_initial_delay
        while True:
            attempt += 1
            ordering = self._ordering
            try:
                self._client.upsert(
                    collection_name=target_collection,
                    points=points,
                    wait=self._wait_for_upsert,
                    ordering=ordering,
                )
                logger.info(
                    "Upsert batch=%d succeeded; collection=%s bytes~=%.0f",
                    len(points),
                    target_collection,
                    estimated_bytes,
                )
                return
            except Exception as exc:
                if ordering is not None and _is_ordering_type_error(exc):
                    logger.warning(
                        "Write ordering preference '%s' rejected by client; retrying without explicit ordering",
                        self._ordering_preference,
                    )
                    self._ordering = None
                    continue
                is_timeout = _is_timeout_error(exc)
                if not is_timeout or attempt >= self._max_retry_attempts:
                    raise
                sleep_for = min(
                    self._retry_max_delay,
                    delay + random.random(),
                )
                logger.warning(
                    "Upsert attempt %d/%d failed with timeout; retrying in %.2fs (points=%d bytes~=%.0f)",
                    attempt,
                    self._max_retry_attempts,
                    sleep_for,
                    len(points),
                    estimated_bytes,
                )
                time.sleep(sleep_for)
                delay = min(delay * 2, self._retry_max_delay)

    def _estimate_point_bytes(
        self,
        title_vector: Sequence[float] | None,
        body_vector: Sequence[float] | None,
        payload: Mapping[str, object],
    ) -> float:
        title_length = len(title_vector or [])
        body_length = len(body_vector or [])
        vector_bytes = 4 * (title_length + body_length)
        payload_bytes = _estimate_payload_bytes(payload)
        return vector_bytes + payload_bytes + 256

    def _estimate_points_bytes(
        self, points: Sequence[rest.PointStruct], vector_dim: int
    ) -> float:
        total = 0.0
        for point in points:
            vectors = getattr(point, "vector", {}) or {}
            title_vec = vectors.get("title_vec") if isinstance(vectors, Mapping) else None
            body_vec = vectors.get("body_vec") if isinstance(vectors, Mapping) else None
            payload = getattr(point, "payload", {}) or {}
            total += self._estimate_point_bytes(title_vec, body_vec, payload)
        return max(total, float(vector_dim))

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

    def _provision_collection(
        self, alias: str, vector_size: int, *, on_disk: bool = True
    ) -> str:
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
                    optimizers_config=_make_optimizer_config(on_disk=on_disk),
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
        try:
            response = self._client.get_aliases()
        except Exception as exc:
            logger.debug("Failed to resolve alias %s: %s", alias, exc)
            return None
        aliases = getattr(response, "aliases", None) or []
        for description in aliases:
            name = getattr(description, "alias_name", None)
            if name == alias:
                collection = getattr(description, "collection_name", None)
                if isinstance(collection, str):
                    return collection
        return None

    def _swap_alias(
        self, alias: str, new_collection: str, previous: Optional[str]
    ) -> bool:
        operations: List[dict[str, dict[str, str]]] = [
            {"create_alias": {"alias_name": alias, "collection_name": new_collection}}
        ]
        if previous and previous != new_collection:
            operations.append(
                {
                    "delete_alias": {
                        "alias_name": alias,
                        "collection_name": previous,
                    }
                }
            )

        try:
            if hasattr(self._client, "update_aliases"):
                self._client.update_aliases(
                    change_aliases_operations=operations
                )
            else:
                self._client.update_collection_aliases(
                    change_aliases_operations=operations
                )
        except Exception as exc:
            logger.warning(
                "Failed to update alias %s -> %s: %s. Using direct collection writes.",
                alias,
                new_collection,
                exc,
            )
            self._write_collection_name = new_collection
            return False

        logger.info("Alias %s now points to collection %s", alias, new_collection)
        self._write_collection_name = self._alias_name
        return True

    def _delete_collection(self, collection: str) -> None:
        try:
            self._client.delete_collection(collection_name=collection)
        except Exception as exc:
            logger.warning("Failed to delete legacy collection %s: %s", collection, exc)

    def swap_alias_atomically(self, alias: str, new_collection: str) -> None:
        previous = self._resolve_alias(alias)
        swapped = self._swap_alias(alias, new_collection, previous)
        if swapped:
            logger.info("Alias swap OK -> %s -> %s", alias, new_collection)
        else:
            logger.warning(
                "Alias swap for %s fell back to direct writes; manual verification recommended",
                alias,
            )

    def cleanup_old_collections(self, alias: str, keep_n: int = 2) -> None:
        current = self._resolve_alias(alias)
        try:
            response = self._client.get_collections()
        except Exception as exc:
            logger.warning("Cleanup skipped: unable to list collections (%s)", exc)
            return
        collections = getattr(response, "collections", None) or []
        names: List[str] = []
        for description in collections:
            name = getattr(description, "name", None) or getattr(
                description, "collection_name", None
            )
            if isinstance(name, str):
                names.append(name)
        relevant = [name for name in names if name.startswith(alias)]
        relevant.sort(reverse=True)
        if current and current in relevant:
            relevant.remove(current)
        stale = relevant[keep_n - 1 :] if keep_n > 0 else relevant
        if stale:
            logger.info(
                "Identified %d stale collection(s) for alias %s: %s",
                len(stale),
                alias,
                ", ".join(stale),
            )
            logger.info("TODO: safe removal to be implemented when retention policy finalised.")


    def _is_collection_missing_error(self, exc: Exception) -> bool:
        """Best-effort detection for missing collection errors across client modes."""

        if isinstance(exc, _ResponseHandlingException):
            status_code = getattr(exc, "status_code", None)
            if status_code == 404:
                return True
        # gRPC errors expose a callable ``code`` that returns a StatusCode enum.
        code_attr = getattr(exc, "code", None)
        try:
            code_value = code_attr() if callable(code_attr) else code_attr
        except Exception:  # pragma: no cover - defensive, unlikely
            code_value = None
        if code_value is not None:
            name = getattr(code_value, "name", "").lower()
            if name == "not_found":
                return True
            if str(code_value).upper().endswith("NOT_FOUND"):
                return True
        # Fall back to string matching to catch REST clients and older SDKs.
        details = getattr(exc, "details", None)
        message = (details or str(exc) or "").lower()
        for needle in ("not found", "doesn't exist", "does not exist"):
            if needle in message:
                return True
        return False

    def count_points(self, collection: str | None = None) -> int:
        target = collection or self.collection_name
        try:
            response = self._client.count(
                collection_name=target, exact=True
            )
        except Exception as exc:
            if self._is_collection_missing_error(exc):
                logger.info(
                    "Collection %s not found during count; treating as empty", target
                )
                return 0
            logger.exception("Failed to count points for collection %s", target)
            raise
        total = int(getattr(response, "count", 0))
        logger.info("Counted %d point(s) in collection %s", total, target)
        return total

    def count_points_for_doc_ids(
        self, collection: str, doc_ids: Sequence[str]
    ) -> Dict[str, int]:
        if not doc_ids:
            return {}
        unique_ids = []
        seen: set[str] = set()
        for doc_id in doc_ids:
            if not doc_id:
                continue
            if doc_id in seen:
                continue
            seen.add(doc_id)
            unique_ids.append(doc_id)
        results: Dict[str, int] = {doc_id: 0 for doc_id in unique_ids}
        for doc_id in unique_ids:
            condition = rest.FieldCondition(
                key="doc_id", match=rest.MatchValue(value=doc_id)
            )
            flt = rest.Filter(must=[condition])
            try:
                response = self._client.count(
                    collection_name=collection,
                    exact=True,
                    filter=flt,
                )
            except Exception as exc:
                if self._is_collection_missing_error(exc):
                    results[doc_id] = 0
                    continue
                logger.warning(
                    "Doc-level count failed for %s in %s: %s",
                    doc_id,
                    collection,
                    exc,
                )
                continue
            results[doc_id] = int(getattr(response, "count", 0))
        return results

    def wait_for_count(self, collection: str, expected: int) -> int:
        if expected <= 0:
            return self.count_points(collection)
        deadline = time.time() + self._validation_max_wait_seconds
        poll_interval = self._validation_poll_interval_seconds
        attempt = 0
        latest = 0
        while True:
            attempt += 1
            latest = self.count_points(collection)
            if latest >= expected:
                if latest > expected:
                    logger.warning(
                        "Validation count exceeded expected total for %s: expected=%d actual=%d",
                        collection,
                        expected,
                        latest,
                    )
                else:
                    logger.info(
                        "Validation count reached expected total for %s on attempt %d",
                        collection,
                        attempt,
                    )
                return latest
            now = time.time()
            if now >= deadline:
                logger.error(
                    "Validation wait timed out for %s after %d attempt(s); last count=%d expected=%d",
                    collection,
                    attempt,
                    latest,
                    expected,
                )
                return latest
            remaining = deadline - now
            sleep_for = min(poll_interval, remaining)
            logger.info(
                "Validation wait for %s attempt=%d count=%d expected=%d sleeping=%.2fs",
                collection,
                attempt,
                latest,
                expected,
                sleep_for,
            )
            time.sleep(max(0.1, sleep_for))

    def health_probe(
        self, collection: str, sample_texts: list[str], limit: int = 5
    ) -> bool:
        try:
            total = self.count_points(collection)
        except Exception:
            logger.error("Health probe failed during count for %s", collection)
            return False
        if total <= 0:
            logger.warning("Health probe: collection %s is empty", collection)
            return False
        try:
            points, _ = self._client.scroll(
                collection_name=collection,
                limit=max(1, limit),
                with_payload=True,
            )
        except Exception:
            logger.exception("Health probe scroll failed for %s", collection)
            return False
        if not points:
            logger.warning("Health probe: no points returned when scrolling %s", collection)
            return False
        lowered = [text.lower() for text in sample_texts if isinstance(text, str) and text]
        matches = 0
        if lowered:
            for point in points:
                payload = getattr(point, "payload", {}) or {}
                blob = " ".join(
                    str(payload.get(key, "")) for key in ("title_text", "body_text")
                ).lower()
                if any(sample in blob for sample in lowered):
                    matches += 1
        logger.info(
            "Health probe for %s matched %d/%d samples (limit=%d total=%d)",
            collection,
            matches,
            len(lowered),
            limit,
            total,
        )
        return bool(points) and (matches > 0 or not lowered)

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
        for path in ("/telemetry", "/version", "/readyz", "/healthz"):
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


def _is_ordering_type_error(exc: Exception) -> bool:
    message = str(exc)
    lowered = message.lower()
    if isinstance(exc, TypeError) and ("writeordering" in lowered or "ordering" in lowered):
        return True
    if "writeordering" in lowered or "upsertpoints.ordering" in lowered:
        return True
    cause = getattr(exc, "__cause__", None)
    if cause and _is_ordering_type_error(cause):
        return True
    context = getattr(exc, "__context__", None)
    if context and _is_ordering_type_error(context):
        return True
    return False


def _is_timeout_error(exc: Exception | None) -> bool:
    if exc is None:
        return False
    timeout_types = (httpx.TimeoutException,)
    if _ResponseHandlingException:
        timeout_types = timeout_types + tuple(
            _ResponseHandlingException
            if isinstance(_ResponseHandlingException, tuple)
            else (_ResponseHandlingException,)
        )
    if isinstance(exc, timeout_types):
        return "timeout" in str(exc).lower()
    message = str(exc).lower()
    if "timeout" in message:
        return True
    cause = getattr(exc, "__cause__", None)
    if cause and _is_timeout_error(cause):
        return True
    context = getattr(exc, "__context__", None)
    if context and _is_timeout_error(context):
        return True
    return False


def _truncate_text(value: object, limit: int) -> str:
    text = value if isinstance(value, str) else (str(value) if value is not None else "")
    if limit <= 0:
        return text
    if len(text) <= limit:
        return text
    return text[:limit]


def _estimate_payload_bytes(payload: Mapping[str, object]) -> float:
    total = 0
    stack: List[object] = [payload]
    while stack:
        item = stack.pop()
        if isinstance(item, Mapping):
            for key, value in item.items():
                total += len(str(key).encode("utf-8"))
                if value is not None:
                    stack.append(value)
        elif isinstance(item, (list, tuple, set)):
            stack.extend(item)
        elif isinstance(item, (bytes, bytearray)):
            total += len(item)
        elif item is None:
            continue
        else:
            total += len(str(item).encode("utf-8"))
    return float(total)


def _slim_payload_if_needed(
    payload: Mapping[str, object], *, enable_slim: bool
) -> Mapping[str, object]:
    if not enable_slim or not isinstance(payload, Mapping):
        return payload
    body_text = payload.get("body_text") if isinstance(payload, Mapping) else None
    if isinstance(body_text, str) and len(body_text) > 2048:
        clone = dict(payload)
        clone.setdefault("body_text_full", body_text)
        clone["body_text"] = body_text[:2048]
        return clone
    return payload


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


def _make_optimizer_config(*, on_disk: bool = True) -> rest.OptimizersConfigDiff:
    threshold = 20000 if on_disk else 0
    return rest.OptimizersConfigDiff(memmap_threshold=threshold)


def _date_string_to_int(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        dt = datetime.strptime(text[:10], "%Y-%m-%d")
    except ValueError:
        digits = "".join(ch for ch in text if ch.isdigit())
        if len(digits) >= 8:
            try:
                return int(digits[:8])
            except ValueError:
                return None
        return None
    return int(dt.strftime("%Y%m%d"))


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


