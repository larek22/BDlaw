from __future__ import annotations

from types import SimpleNamespace

from app.query_pipeline import QueryPipeline
from app.settings import AppSettings


def test_law_filters_apply_date_and_status(monkeypatch):
    settings = AppSettings()
    settings.query.enable_law_filters = True
    settings.query.status_filter = "active"
    settings.query.as_of_date = "2024-02-01"
    settings.query.part_no = 2

    monkeypatch.setattr("app.query_pipeline.build_reranker", lambda *args, **kwargs: None)
    monkeypatch.setattr("app.query_pipeline.OpenAI", lambda *args, **kwargs: SimpleNamespace())

    pipeline = object.__new__(QueryPipeline)
    pipeline.settings = settings
    pipeline.embedding_client = SimpleNamespace()
    pipeline.vector_store = SimpleNamespace()

    filter_ = QueryPipeline._build_law_filters(pipeline)
    assert filter_ is not None
    keys = {condition.key for condition in filter_.must}
    assert "law_meta.status" in keys
    assert "law_meta.date_from_int" in keys
    assert "law_meta.date_to_int" in keys
    assert "hierarchy.part_no" in keys
