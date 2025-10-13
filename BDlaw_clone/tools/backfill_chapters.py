#!/usr/bin/env python
"""Backfill missing chapter metadata for existing Qdrant points."""

from __future__ import annotations

import argparse
import logging
import os
from typing import Dict, Iterable, Tuple

from qdrant_client import QdrantClient

from app.legal_pipeline import _ARTICLE_RE, _CHAPTER_RE
from app.utils.net import normalize_qdrant_url

logger = logging.getLogger(__name__)


def _load_mapping(text: str) -> Dict[int, Tuple[int, str | None]]:
    mapping: Dict[int, Tuple[int, str | None]] = {}
    chapter_no: int | None = None
    chapter_title: str | None = None

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        chapter_match = _CHAPTER_RE.match(stripped)
        if chapter_match:
            try:
                chapter_no = int(chapter_match.group(1))
            except (TypeError, ValueError):
                chapter_no = None
            chapter_title = chapter_match.group(2).strip() or None
            continue
        article_match = _ARTICLE_RE.match(stripped)
        if article_match and chapter_no is not None:
            raw_article = article_match.group(1)
            base = raw_article.split(".", 1)[0]
            if base.isdigit():
                mapping[int(base)] = (chapter_no, chapter_title)
    return mapping


def _gather_updates(
    client: QdrantClient, collection: str, mapping: Dict[int, Tuple[int, str | None]]
) -> Iterable[Tuple[str, dict]]:
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
            if chapter not in (None, ""):
                continue
            article = hierarchy.get("article_no")
            if not isinstance(article, str):
                continue
            key = article.split(".", 1)[0]
            if not key.isdigit():
                continue
            article_base = int(key)
            if article_base not in mapping:
                continue
            new_chapter, new_title = mapping[article_base]
            update_payload = {
                "hierarchy.chapter_no": new_chapter,
            }
            if new_title:
                update_payload["hierarchy.chapter_title"] = new_title
            point_id = getattr(point, "id", None)
            if point_id is not None:
                yield str(point_id), update_payload
        if not offset:
            break


def backfill(url: str, api_key: str | None, collection: str, normalized_path: str) -> None:
    logging.basicConfig(level=logging.INFO)
    logger.info("Loading normalized text from %s", normalized_path)
    with open(normalized_path, "r", encoding="utf-8") as handle:
        text = handle.read()
    mapping = _load_mapping(text)
    if not mapping:
        raise SystemExit("Failed to derive chapter mapping from normalized text")

    client = QdrantClient(url=url, api_key=api_key or None, prefer_grpc=False)
    updates_applied = 0
    for point_id, payload in _gather_updates(client, collection, mapping):
        client.set_payload(collection_name=collection, payload=payload, points=[point_id])
        updates_applied += 1

    if updates_applied == 0:
        logger.info("No chapter backfill required")
        return

    logger.info("Backfill applied to %d point(s)", updates_applied)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill chapter metadata in Qdrant")
    parser.add_argument("--normalized-path", default=os.getenv("NORMALIZED_PATH", "data/users.normalized.txt"))
    args = parser.parse_args()

    url = normalize_qdrant_url(os.getenv("QDRANT_URL"))
    api_key = os.getenv("QDRANT_API_KEY")
    collection = os.getenv("QDRANT_COLLECTION", "kb_docs_v1")
    backfill(url, api_key, collection, args.normalized_path)


if __name__ == "__main__":  # pragma: no cover
    main()
