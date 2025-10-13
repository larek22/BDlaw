from __future__ import annotations

import pytest

from app.ingest import IngestStats
from app.settings import AppSettings
from tests_helpers_ingest import (
    FakeVectorStore,
    build_service,
    run_atomic,
)


def test_atomic_alias_not_swapped_on_validation_failure(monkeypatch):
    settings = AppSettings()
    settings.ingest.atomic_alias_swap = True
    vector_store = FakeVectorStore()
    service = build_service(settings, vector_store)

    stats = run_atomic(service, vector_store, count_override=0, health_ok=True)

    assert stats.chunks_created == 1
    assert vector_store.swapped is False


def test_atomic_alias_swapped_on_success(monkeypatch):
    settings = AppSettings()
    settings.ingest.atomic_alias_swap = True
    vector_store = FakeVectorStore()
    service = build_service(settings, vector_store)

    stats = run_atomic(service, vector_store, count_override=1, health_ok=True)

    assert stats.chunks_created == 1
    assert vector_store.swapped is True
    assert vector_store.cleaned is True
    assert vector_store.last_swap == (vector_store.alias_name, vector_store.shadow_collection)
