from pathlib import Path

from app.settings import AppSettings

from tests.test_atomic_alias_swap import (
    StubVectorStore,
    _make_service,
)


def test_reingest_is_idempotent():
    settings = AppSettings()
    settings.ingest.atomic_alias_swap = True
    settings.ingest.validation_sample_k = 2
    vector_store = StubVectorStore(healthy=True)
    service = _make_service(settings, vector_store)

    service.ingest([Path("/tmp/doc1.pdf")])
    service.ingest([Path("/tmp/doc1.pdf")])

    assert vector_store.swap_calls == 2
    assert vector_store.active_count == 1
    assert vector_store.count_points(vector_store.shadow_collection) == 1
