from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Iterable, List, Sequence

try:  # pragma: no cover - optional during tests
    from qdrant_client import QdrantClient  # type: ignore
    from qdrant_client.http.models import (  # type: ignore
        Distance,
        VectorParams,
        PayloadSchemaType,
    )
except Exception:  # pragma: no cover
    QdrantClient = None  # type: ignore

    class Distance:  # type: ignore
        COSINE = "Cosine"

    class VectorParams:  # type: ignore
        def __init__(self, size: int, distance: str) -> None:
            self.size = size
            self.distance = distance

    class PayloadSchemaType:  # type: ignore
        KEYWORD = "keyword"
        INTEGER = "integer"

try:  # pragma: no cover
    from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception_type
except Exception:  # pragma: no cover
    retry = stop_after_attempt = wait_exponential_jitter = retry_if_exception_type = None  # type: ignore

from .id_utils import point_id_from_sha
from .settings import IngestRuntime

LOGGER = logging.getLogger(__name__)


class QdrantWriteTimeout(RuntimeError):
    pass


class SafeQdrantClient:
    """A conservative façade over :class:`qdrant_client.QdrantClient`."""

    def __init__(self, settings: IngestRuntime | None = None) -> None:
        self.settings = settings or IngestRuntime()
        if QdrantClient is None:
            raise RuntimeError("qdrant-client must be installed to use SafeQdrantClient")
        cfg = self.settings.qdrant
        self._client = QdrantClient(
            url=cfg.url,
            api_key=cfg.api_key,
            prefer_grpc=cfg.prefer_grpc,
            grpc_port=cfg.grpc_port,
            timeout=cfg.read_timeout,
            https=cfg.tls,
        )

    def create_physical_collection(self, vector_dim: int) -> str:
        cfg = self.settings.qdrant
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        name = f"{cfg.collection_prefix}_{timestamp}"
        LOGGER.info("Creating Qdrant collection %s (dim=%s)", name, vector_dim)
        self._client.create_collection(
            collection_name=name,
            vectors_config={
                "title_vec": VectorParams(size=vector_dim, distance=Distance.COSINE),
                "body_vec": VectorParams(size=vector_dim, distance=Distance.COSINE),
            },
        )
        return name

    def ensure_payload_indexes(self, collection_name: str) -> None:
        indexes = {
            "doc_id": PayloadSchemaType.KEYWORD,
            "hierarchy.part_no": PayloadSchemaType.INTEGER,
            "hierarchy.chapter_no": PayloadSchemaType.INTEGER,
            "hierarchy.article_no_int": PayloadSchemaType.INTEGER,
            "hierarchy.section_roman": PayloadSchemaType.KEYWORD,
            "law_meta.status": PayloadSchemaType.KEYWORD,
            "law_meta.enact_date_int": PayloadSchemaType.INTEGER,
            "law_meta.last_amend_date_int": PayloadSchemaType.INTEGER,
            "law_meta.date_from_int": PayloadSchemaType.INTEGER,
            "law_meta.date_to_int": PayloadSchemaType.INTEGER,
        }
        for field, schema in indexes.items():
            try:
                self._client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field,
                    field_schema=schema,
                )
            except Exception as exc:  # noqa: BLE001
                if "already exists" not in str(exc).lower():
                    raise

    def finalize_collection(self, alias: str, new_collection: str) -> None:
        LOGGER.info("Swapping alias %s -> %s", alias, new_collection)
        current = self._resolve_alias(alias)
        ops: List[dict] = [
            {"create_alias": {"alias_name": alias, "collection_name": new_collection}}
        ]
        if current:
            ops.append({"delete_alias": {"alias_name": alias, "collection_name": current}})
        self._client.update_aliases(change_aliases_operations=ops)

    def _resolve_alias(self, alias: str) -> str | None:
        try:
            response = self._client.get_aliases(alias)
        except Exception:
            return None
        collections = getattr(response, "collections", None)
        if collections:
            return collections[0]
        return None

    def upsert_points(
        self,
        collection_name: str,
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[dict],
        shard_key: str | None = None,
    ) -> None:
        if len(vectors) != len(payloads):
            raise ValueError("vectors and payloads length mismatch")
        cfg = self.settings.qdrant
        points = [
            {
                "id": point_id_from_sha(payload["sha256"]),
                "vector": {"title_vec": vectors[idx], "body_vec": vectors[idx]},
                "payload": payload,
            }
            for idx, payload in enumerate(payloads)
        ]
        for batch in _batch_points(points, cfg.target_batch_bytes, cfg.max_points_per_batch):
            self._commit_with_retry(collection_name, batch, shard_key)

    def _commit_with_retry(self, collection: str, batch: Sequence[dict], shard_key: str | None) -> None:
        if retry is None:
            raise RuntimeError("tenacity must be installed to use SafeQdrantClient")
        cfg = self.settings.qdrant

        @retry(  # type: ignore[misc]
            reraise=True,
            stop=stop_after_attempt(cfg.max_retries),
            wait=wait_exponential_jitter(initial=1, max=15),
            retry=retry_if_exception_type(QdrantWriteTimeout),
        )
        def _inner() -> None:
            try:
                self._client.upsert(
                    collection_name=collection,
                    points=batch,
                    wait=cfg.wait_for_upsert,
                    ordering=cfg.ordering,
                    shard_key=shard_key,
                )
            except Exception as exc:  # noqa: BLE001
                if "timeout" in str(exc).lower():
                    raise QdrantWriteTimeout(str(exc)) from exc
                raise

        _inner()


def _batch_points(points: Sequence[dict], target_bytes: int, max_points: int) -> Iterable[List[dict]]:
    batch: List[dict] = []
    est_bytes = 0
    for point in points:
        point_bytes = _estimate_point_size(point)
        if batch and (est_bytes + point_bytes > target_bytes or len(batch) >= max_points):
            yield batch
            batch = []
            est_bytes = 0
        batch.append(point)
        est_bytes += point_bytes
    if batch:
        yield batch


def _estimate_point_size(point: dict) -> int:
    vector = point["vector"].get("body_vec", [])
    vector_bytes = len(vector) * 4
    payload_bytes = sum(len(str(value)) for value in point["payload"].values())
    return vector_bytes * 2 + payload_bytes + 512


__all__ = ["SafeQdrantClient", "QdrantWriteTimeout"]
