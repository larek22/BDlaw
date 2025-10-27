#!/usr/bin/env python
"""Post-ingestion verification checks for the legal Qdrant collection."""

from __future__ import annotations

import logging
import os
import sys
from typing import Dict

from qdrant_client import QdrantClient

from app.utils.net import normalize_qdrant_url

logger = logging.getLogger(__name__)


def _count_points(client: QdrantClient, collection: str) -> int:
    try:
        return client.count(collection_name=collection, exact=True).count
    except Exception as exc:  # pragma: no cover - network failure path
        logger.error("Failed to count points: %s", exc)
        raise


def _iterate_points(client: QdrantClient, collection: str):
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=True,
        )
        if not points:
            break
        for point in points:
            yield point
        if not offset:
            break


def verify() -> int:
    logging.basicConfig(level=logging.INFO)

    url = normalize_qdrant_url(os.getenv("QDRANT_URL"))
    api_key = os.getenv("QDRANT_API_KEY")
    collection = os.getenv("QDRANT_COLLECTION", "kb_docs_v1")
    expected_points_env = os.getenv("EXPECTED_POINTS")
    smoke_chapter = os.getenv("SMOKE_CHAPTER")

    expected_points = int(expected_points_env) if expected_points_env else None

    client = QdrantClient(url=url, api_key=api_key or None, prefer_grpc=False)

    total = _count_points(client, collection)
    logger.info("Collection %s contains %d point(s)", collection, total)
    if expected_points is not None and total != expected_points:
        logger.warning(
            "Total point count mismatch: expected %d but found %d", expected_points, total
        )

    missing = 0
    chapters: Dict[int, int] = {}
    for point in _iterate_points(client, collection):
        payload = getattr(point, "payload", {}) or {}
        hierarchy = payload.get("hierarchy") or {}
        if not isinstance(hierarchy, dict):
            continue
        chapter = hierarchy.get("chapter_no")
        if chapter in (None, ""):
            missing += 1
        elif isinstance(chapter, int):
            chapters[chapter] = chapters.get(chapter, 0) + 1

    if missing:
        logger.error("Found %d point(s) without chapter metadata", missing)
        return 1

    if smoke_chapter:
        try:
            chapter_id = int(smoke_chapter)
        except ValueError:
            chapter_id = None
        if chapter_id is not None and chapters.get(chapter_id, 0) == 0:
            logger.error("Smoke chapter %d has no points", chapter_id)
            return 1

    logger.info("Chapter distribution: %s", chapters)
    logger.info("Verification completed successfully")
    return 0


def main() -> None:
    sys.exit(verify())


if __name__ == "__main__":  # pragma: no cover
    main()
