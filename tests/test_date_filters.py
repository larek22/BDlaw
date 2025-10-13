from __future__ import annotations

from types import SimpleNamespace

from app.query_pipeline import QueryPipeline
from app.settings import AppSettings
from app.qdrant_client import QdrantVectorStore
from qdrant_client.http import models as rest


def test_as_of_filter_includes_date_ranges():
    store = object.__new__(QdrantVectorStore)
    result = store.build_as_of_filter(status="in_force", as_of_date="2024-01-01")
    keys = {condition.key for condition in result.must}
    assert "law_meta.status" in keys
    if hasattr(rest, "Range"):
        assert "law_meta.date_from_int" in keys
        assert "law_meta.date_to_int" in keys


class DummyVectorStore:
    def keyword_prefilter(self, query: str, limit: int):
        return []

    def build_doc_id_filter(self, doc_ids):  # pragma: no cover - not exercised
        raise NotImplementedError

    def build_as_of_filter(self, **kwargs):
        return rest.Filter(must=[rest.FieldCondition(key="law_meta.status", match=rest.MatchValue(value="in_force"))])

    def combine_filters(self, *filters):
        return rest.Filter(must=[condition for flt in filters if flt for condition in flt.must])

    def query(self, *args, **kwargs):  # pragma: no cover - unused
        raise NotImplementedError


class DummyEmbeddingClient:
    def embed_query(self, text: str):  # pragma: no cover - unused
        raise NotImplementedError


def test_query_pipeline_hierarchy_filters():
    settings = AppSettings()
    settings.query.enable_law_filters = True
    settings.query.filter_part_no = 2
    settings.query.filter_chapter_no = 5
    settings.query.filter_article_no_int = 783
    pipeline = QueryPipeline(settings, DummyEmbeddingClient(), DummyVectorStore())
    flt = pipeline._build_combined_filter("question", top_k=1)
    keys = [condition.key for condition in flt.must]
    assert "hierarchy.part_no" in keys
    assert "hierarchy.chapter_no" in keys
    assert "hierarchy.article_no_int" in keys
