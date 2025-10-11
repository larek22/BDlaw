#!/usr/bin/env python3
"""Backfill optional payload fields for existing Qdrant points."""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, Tuple

from app.logging_config import configure_logging
from app.qdrant_client import QdrantVectorStore
from app.settings import AppSettings

logger = logging.getLogger(__name__)

_DEFAULT_PLAN_VERSION = "planner.legacy"
_DEFAULT_PARSER_VERSION = "parser.legacy"


def _coerce_int(value: Any) -> Tuple[bool, int | None]:
    """Attempt to coerce *value* into an integer."""

    if isinstance(value, int):
        return True, value
    if isinstance(value, str) and value.strip():
        try:
            return True, int(value)
        except ValueError:
            return False, None
    return False, None


def _safe_text(value: Any, default: str = "") -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return default
    return str(value)


def _compute_chunk_key(payload: Dict[str, Any]) -> str | None:
    doc_id = _safe_text(payload.get("doc_id"))
    _, chunk_index = _coerce_int(payload.get("chunk_index"))
    body_text = _safe_text(payload.get("body_text"))
    if not doc_id or chunk_index is None:
        return None
    if not body_text:
        return None
    seed = f"{doc_id}:{chunk_index}:{body_text[:64]}"
    return hashlib.sha1(seed.encode("utf-8")).hexdigest()


def migrate() -> None:
    configure_logging()
    settings = AppSettings.load()
    store = QdrantVectorStore(settings)

    collection = store.collection_name
    logger.info("Starting payload backfill for collection %s", collection)

    total = 0
    updated = 0
    skipped = 0
    offset = None

    while True:
        points, offset = store.client.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=True,
        )
        if not points:
            break
        for point in points:
            total += 1
            payload = getattr(point, "payload", {}) or {}
            point_id = getattr(point, "id", None)
            updates: Dict[str, Any] = {}

            if not point_id:
                skipped += 1
                continue

            has_index, chunk_index = _coerce_int(payload.get("chunk_index"))
            if not has_index:
                logger.debug(
                    "Skipping point %s: unable to determine chunk_index", point_id
                )
                skipped += 1
                continue

            if not _safe_text(payload.get("title_text")):
                updates["title_text"] = ""

            if "chunk_key" not in payload or not _safe_text(payload.get("chunk_key")):
                chunk_key = _compute_chunk_key(payload)
                if chunk_key:
                    updates["chunk_key"] = chunk_key

            if "plan_version" not in payload or not _safe_text(payload.get("plan_version")):
                updates["plan_version"] = _DEFAULT_PLAN_VERSION

            if "parser_version" not in payload or not _safe_text(payload.get("parser_version")):
                updates["parser_version"] = _DEFAULT_PARSER_VERSION

            if not updates:
                continue

            try:
                store.client.set_payload(
                    collection_name=collection,
                    payload=updates,
                    points=[point_id],
                )
            except Exception as exc:  # pragma: no cover - network failure path
                logger.warning("Failed to update point %s: %s", point_id, exc)
                skipped += 1
            else:
                updated += 1

        if offset is None:
            break

    logger.info(
        "Payload backfill completed: %d point(s) inspected, %d updated, %d skipped",
        total,
        updated,
        skipped,
    )


def main() -> None:
    migrate()


if __name__ == "__main__":  # pragma: no cover
    main()
