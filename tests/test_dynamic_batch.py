from __future__ import annotations

from app.qdrant_client import QdrantVectorStore
from app.settings import AppSettings


class FakeClient:
    def __init__(self, **kwargs) -> None:
        self.calls: list[int] = []

    def upsert(self, collection_name, points, wait, ordering):
        self.calls.append(len(points))
        if len(self.calls) == 1:
            raise TimeoutError("timeout")


def test_dynamic_batch(monkeypatch):
    settings = AppSettings()
    settings.qdrant.prefer_grpc = False
    settings.qdrant.url = "http://localhost:6333"
    settings.qdrant.api_key = None
    settings.qdrant.max_batch_size = 4
    settings.qdrant.min_batch_size = 2

    fake_client = FakeClient()

    def fake_client_factory(**kwargs):
        return fake_client

    monkeypatch.setattr("app.qdrant_client.QdrantClient", fake_client_factory)
    monkeypatch.setattr(QdrantVectorStore, "_fetch_server_version", lambda self: "test")

    store = QdrantVectorStore(settings)
    points = [object() for _ in range(4)]
    store._upsert_batch(points, vector_dim=3, estimated_bytes=128.0, collection_name="shadow")
    assert fake_client.calls == [4, 2, 2]
