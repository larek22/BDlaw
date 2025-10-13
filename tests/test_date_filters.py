from __future__ import annotations

from qdrant_client.http import models as rest

from app.qdrant_client import QdrantVectorStore


def test_build_law_filters_adds_expected_conditions():
    store = object.__new__(QdrantVectorStore)
    filter_ = QdrantVectorStore.build_law_filters(
        store,
        as_of_date="2024-01-01",
        status="in_force",
        part_no=1,
        chapter_no=2,
        article_no_int=783,
    )

    assert filter_ is not None
    keys = {getattr(cond, "key", None) for cond in filter_.must}
    assert "law_meta.status" in keys
    assert "hierarchy.part_no" in keys
    assert "hierarchy.chapter_no" in keys
    assert "hierarchy.article_no_int" in keys

    ranges = [getattr(cond, "range", None) for cond in filter_.must if getattr(cond, "range", None)]
    assert any(getattr(rng, "lte", None) == 20240101 for rng in ranges)
    assert any(getattr(rng, "gte", None) == 20240101 for rng in ranges)
