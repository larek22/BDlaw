#!/usr/bin/env python
"""Print simple statistics about the current Qdrant collection."""

from __future__ import annotations

import logging
import os

from qdrant_client import QdrantClient

from app.utils.net import normalize_qdrant_url

logger = logging.getLogger(__name__)


def dump_stats() -> None:
    logging.basicConfig(level=logging.INFO)

    url = normalize_qdrant_url(os.getenv("QDRANT_URL"))
    api_key = os.getenv("QDRANT_API_KEY")
    collection = os.getenv("QDRANT_COLLECTION", "kb_docs_v1")

    client = QdrantClient(url=url, api_key=api_key or None, prefer_grpc=False)
    total = client.count(collection_name=collection, exact=True).count
    logger.info("Collection %s contains %d point(s)", collection, total)

    chapter_counts: dict[int, int] = {}
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
            payload = getattr(point, "payload", {}) or {}
            hierarchy = payload.get("hierarchy") or {}
            if not isinstance(hierarchy, dict):
                continue
            chapter = hierarchy.get("chapter_no")
            if isinstance(chapter, int):
                chapter_counts[chapter] = chapter_counts.get(chapter, 0) + 1
        if not offset:
            break

    for chapter in sorted(chapter_counts):
        logger.info("Chapter %d: %d point(s)", chapter, chapter_counts[chapter])


if __name__ == "__main__":  # pragma: no cover
    dump_stats()
