from __future__ import annotations

import pytest

from app.qdrant_client import SafeQdrantClient
from app.settings import IngestRuntime, QdrantConfig


class DummyClient:
    def __init__(self, **kwargs) -> None:
        self.collection_args = None
        self.upsert_calls = []
        self.alias_ops = []

    def create_collection(self, collection_name: str, vectors_config: dict) -> None:
        self.collection_args = (collection_name, vectors_config)

    def create_payload_index(self, *args, **kwargs):
        return None

    def upsert(self, **kwargs):
        self.upsert_calls.append(kwargs)

    def update_aliases(self, change_aliases_operations):
        self.alias_ops.append(change_aliases_operations)

    def get_aliases(self, alias: str):
        class Response:
            collections = []

        return Response()


@pytest.fixture(autouse=True)
def stub_client(monkeypatch):
    dummy = DummyClient()
    monkeypatch.setattr("app.qdrant_client.QdrantClient", lambda **_: dummy)

    def passthrough_retry(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    monkeypatch.setattr("app.qdrant_client.retry", passthrough_retry)
    monkeypatch.setattr("app.qdrant_client.stop_after_attempt", lambda *_, **__: None)
    monkeypatch.setattr("app.qdrant_client.wait_exponential_jitter", lambda *_, **__: None)
    monkeypatch.setattr("app.qdrant_client.retry_if_exception_type", lambda *_, **__: None)
    return dummy


def test_create_collection_uses_prefix(stub_client):
    runtime = IngestRuntime(qdrant=QdrantConfig(collection_prefix="TESTCOL"))
    client = SafeQdrantClient(runtime)
    name = client.create_physical_collection(vector_dim=3072)
    assert name.startswith("TESTCOL_")
    assert stub_client.collection_args[1]["body_vec"].size == 3072


def test_upsert_batches_by_size(stub_client):
    runtime = IngestRuntime(qdrant=QdrantConfig(target_batch_bytes=500, max_points_per_batch=4))
    client = SafeQdrantClient(runtime)
    vectors = [[0.0] * 8 for _ in range(10)]
    payloads = [{"sha256": f"{idx:032x}", "title_text": "t", "body_preview": "b"} for idx in range(10)]
    client.upsert_points("collection", vectors, payloads)
    assert len(stub_client.upsert_calls) >= 3


def test_finalize_collection_builds_alias_ops(stub_client):
    runtime = IngestRuntime()
    client = SafeQdrantClient(runtime)
    client.finalize_collection("GK_RF2", "NEWCOL")
    ops = stub_client.alias_ops[-1]
    assert {"create_alias": {"alias_name": "GK_RF2", "collection_name": "NEWCOL"}} in ops
