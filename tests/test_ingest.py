from app.ingest import Chunk, IngestService
from app.settings import IngestRuntime, QdrantConfig


def test_ingest_creates_collection(monkeypatch):
    calls = {}

    class DummyClient:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def create_collection(self, collection_name: str, vectors_config: dict) -> None:
            calls["collection"] = collection_name

        def create_payload_index(self, *args, **kwargs):
            return None

        def upsert(self, **kwargs):
            calls.setdefault("upserts", 0)
            calls["upserts"] += len(kwargs["points"])

        def update_aliases(self, **kwargs):
            pass

        def get_aliases(self, alias: str):
            class Response:
                collections = []

            return Response()

    monkeypatch.setattr("app.qdrant_client.QdrantClient", lambda **_: DummyClient())
    monkeypatch.setattr("app.qdrant_client.retry", lambda *_, **__: (lambda f: f))
    monkeypatch.setattr("app.qdrant_client.stop_after_attempt", lambda *_, **__: None)
    monkeypatch.setattr("app.qdrant_client.wait_exponential_jitter", lambda *_, **__: None)
    monkeypatch.setattr("app.qdrant_client.retry_if_exception_type", lambda *_, **__: None)

    runtime = IngestRuntime(qdrant=QdrantConfig(collection_prefix="TEST"))
    service = IngestService(runtime)
    chunk = Chunk(
        doc_id="doc-1",
        text="text",
        vector=[0.0, 0.1, 0.2, 0.3],
        metadata={"hierarchy": {"part_no": 1}},
    )
    collection = service.ingest(vector_dim=4, chunks=[chunk])
    assert collection.startswith("TEST_")
    assert calls["upserts"] == 1
