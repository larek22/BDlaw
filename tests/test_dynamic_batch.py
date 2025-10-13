from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from app.qdrant_client import QdrantVectorStore
from app.settings import AppSettings


class StubClient:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def upsert(self, collection_name: str, points, wait: bool, ordering: str) -> None:
        self.calls.append(len(points))
        if len(self.calls) == 1:
            raise httpx.WriteTimeout("timeout")


@pytest.mark.parametrize("total_points", [5])
def test_dynamic_upsert_shrinks_batches(monkeypatch, total_points: int) -> None:
    settings = AppSettings()
    settings.qdrant.max_batch_size = 4
    settings.qdrant.min_batch_size = 1
    settings.qdrant.use_wait = False

    store = QdrantVectorStore(settings)
    stub = StubClient()
    store._client = stub  # type: ignore[attr-defined]
    store._max_batch_size = 4
    store._min_batch_size = 1

    monkeypatch.setattr("app.qdrant_client.time.sleep", lambda *_args, **_kwargs: None)

    points = [
        SimpleNamespace(
            id=str(idx),
            vector={"body_vec": [0.1, 0.2], "title_vec": [0.1, 0.2]},
            payload={"doc_id": "doc", "chunk_index": idx},
        )
        for idx in range(total_points)
    ]

    store._upsert_points_dynamic("alias_shadow", points)

    assert stub.calls[0] == 4
    assert stub.calls[-1] == 1
    assert sum(stub.calls) >= total_points
