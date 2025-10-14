from __future__ import annotations

from app.settings import AppSettings
from tests_helpers_ingest import (
    FakeVectorStore,
    build_service,
    make_chunk,
    make_processed,
)


def test_reingest_same_chunks_keeps_point_count():
    settings = AppSettings()
    settings.ingest.atomic_alias_swap = True
    vector_store = FakeVectorStore()
    service = build_service(settings, vector_store)

    chunk = make_chunk("doc-1", 0)
    processed = [make_processed()]
    title_vectors = {chunk.title_sha256: [0.1, 0.2, 0.3, 0.4]}
    body_vectors = {chunk.body_sha256: [0.1, 0.2, 0.3, 0.4]}

    # First ingest creates the staged collection and swaps alias
    vector_store.count_override = None
    vector_store.health_ok = True
    service._ingest_atomic_flow(
        report=lambda *_args, **_kwargs: None,
        processed_documents=processed,
        changed_chunks=[chunk],
        title_vectors=title_vectors,
        body_vectors=body_vectors,
        embedding_model=service.settings.openai_models.embedding,
        skipped=0,
        doc_expected_counts={chunk.doc_id: 1},
    )

    # Re-ingesting the same chunk should keep the point count stable
    vector_store.count_override = None
    service._ingest_atomic_flow(
        report=lambda *_args, **_kwargs: None,
        processed_documents=processed,
        changed_chunks=[make_chunk("doc-1", 0)],
        title_vectors=title_vectors,
        body_vectors=body_vectors,
        embedding_model=service.settings.openai_models.embedding,
        skipped=0,
        doc_expected_counts={chunk.doc_id: 1},
    )

    staged_points = vector_store.points.get(vector_store.shadow_collection, {})
    total_points = sum(len(ids) for ids in staged_points.values())
    assert total_points == 1
