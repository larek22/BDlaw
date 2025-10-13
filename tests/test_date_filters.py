from __future__ import annotations

from app.qdrant_client import QdrantVectorStore
from app.settings import AppSettings


class StubClient:
    def __init__(self, **kwargs) -> None:
        pass


def test_date_filters(monkeypatch):
    settings = AppSettings()
    settings.qdrant.prefer_grpc = False
    settings.qdrant.url = "http://localhost:6333"
    settings.qdrant.api_key = None

    monkeypatch.setattr("app.qdrant_client.QdrantClient", lambda **kwargs: StubClient())
    monkeypatch.setattr(QdrantVectorStore, "_fetch_server_version", lambda self: "test")

    store = QdrantVectorStore(settings)
    hierarchy = {"part_no": 2, "chapter_no": 5, "article_no_int": 7831}
    flt = store.build_law_filter(as_of_date="2024-01-15", in_force_only=True, hierarchy=hierarchy)
    assert flt is not None
    keys = [getattr(cond, "key", "") for cond in getattr(flt, "must", [])]
    assert "law_meta.status" in keys
    assert "law_meta.date_from_int" in keys
    assert "law_meta.date_to_int" in keys
    assert "hierarchy.part_no" in keys
