from __future__ import annotations

import httpx
import pytest

from app.qdrant_client import QdrantVectorStore
from app.settings import AppSettings
from qdrant_client.http import models as rest


class FailingUpsertClient:
    def __init__(self) -> None:
        self.calls: list[int] = []
        self.fail = True

    def upsert(self, *, collection_name, points, wait, ordering):
        self.calls.append(len(points))
        if self.fail:
            self.fail = False
            raise getattr(httpx, "TimeoutException", Exception)("timeout")


@pytest.fixture
def vector_store(monkeypatch) -> QdrantVectorStore:
    settings = AppSettings()
    store = object.__new__(QdrantVectorStore)
    store.settings = settings
    store._client = FailingUpsertClient()
    store._max_batch_size = 8
    store._min_batch_size = 2
    store._retry_initial_delay = 0.0
    store._retry_max_delay = 0.0
    store._max_retry_attempts = 1
    store._wait_for_upsert = False
    store._ordering = "weak"
    monkeypatch.setattr("time.sleep", lambda *_args, **_kwargs: None)
    return store


def test_dynamic_batch_halves_on_timeout(vector_store: QdrantVectorStore):
    points = [rest.PointStruct(id=str(i), vector={}, payload={}) for i in range(6)]
    vector_store._upsert_points_dynamic("alias", points)
    assert vector_store._client.calls[0] == 6
    assert vector_store._client.calls[1] == 3
