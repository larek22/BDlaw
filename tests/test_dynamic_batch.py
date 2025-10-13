from __future__ import annotations

import time

from app.qdrant_client import QdrantVectorStore
from app.settings import AppSettings


class FlakyClient:
    def __init__(self) -> None:
        self.calls: list[int] = []
        self.failures_remaining = 1

    def upsert(self, collection_name, points, wait, ordering):
        self.calls.append(len(points))
        if self.failures_remaining > 0:
            self.failures_remaining -= 1
            raise Exception("timeout")


def test_dynamic_batch_halves_on_timeout(monkeypatch):
    settings = AppSettings()
    settings.qdrant.max_batch_size = 4
    settings.qdrant.min_batch_size = 1
    store = object.__new__(QdrantVectorStore)
    store._client = FlakyClient()
    store._max_batch_size = settings.qdrant.max_batch_size
    store._min_batch_size = settings.qdrant.min_batch_size
    store._use_wait = settings.qdrant.use_wait
    store._ordering = "weak"
    store._retry_initial_delay = 0.0
    store._retry_max_delay = 0.0
    store._wait_for_upsert = False
    store._write_collection_name = "laws_active"

    monkeypatch.setattr(time, "sleep", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("app.qdrant_client.httpx.TimeoutException", Exception, raising=False)
    monkeypatch.setattr("app.qdrant_client.httpx.WriteTimeout", Exception, raising=False)

    store._upsert_points_dynamic([object()] * 4, 3, 0.0, collection_override="shadow")

    assert store._client.calls == [4, 2, 2]
