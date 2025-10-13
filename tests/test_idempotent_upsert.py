from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import uuid

from app.embeddings import EmbeddingResult
from app.ingest import IngestService
from app.legal_types import ChunkRecord
from app.readers.base import DocumentText
from app.settings import AppSettings


@dataclass
class FakeProcessed:
    articles: List[object]
    chunks: List[ChunkRecord]
    doc_ids_changed: List[str]
    doc_ids_unchanged: List[str]


class DummyEmbeddingClient:
    def reset_usage(self) -> None:
        pass

    def embed_texts(self, texts, shas, batch_size=64, max_retries=5):
        return [EmbeddingResult(sha=sha, vector=[0.1, 0.2, 0.3]) for sha in shas]

    def usage_summary(self):
        return {"requests": 1.0, "cache_hits": 0.0, "tokens": 3.0, "cost": 0.0}


class DummyReaderFactory:
    def __init__(self, doc: DocumentText) -> None:
        self._doc = doc

    def read(self, path: Path) -> DocumentText:
        return self._doc


class TrackingVectorStore:
    collection_name = "alias"
    endpoint_url = "http://example"

    def __init__(self) -> None:
        self.alias_name = "alias"
        self.points: set[str] = set()
        self.swap_calls: List[tuple[str, str]] = []
        self.health_ok = True
        self.shadow_name = "alias_shadow"

    def vector_size_for_model(self, model: str) -> int:
        return 3

    def ensure_shadow_collection(self, alias: str, dim: int) -> str:
        self.shadow_name = f"{alias}_shadow"
        return self.shadow_name

    def upsert_chunks(self, chunks, title_vectors, body_vectors, *, collection: str | None = None):
        for chunk in chunks:
            self.points.add(chunk.chunk_id)

    def count_points(self, collection: str | None = None) -> int:
        return len(self.points)

    def health_probe(self, collection: str, sample_texts: list[str], limit: int = 5) -> bool:
        return self.health_ok

    def swap_alias_atomically(self, alias: str, new_collection: str) -> None:
        self.swap_calls.append((alias, new_collection))

    def cleanup_old_collections(self, alias: str, keep_n: int = 2) -> None:
        pass


def create_chunk() -> ChunkRecord:
    chunk = ChunkRecord(
        doc_id="law-001",
        chunk_index=0,
        title_text="Статья 1",
        body_text="Статья 1. Договор наследство.",
        chunk_sha256="chunk",
        title_sha256="title",
        body_sha256="body",
    )
    chunk.chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{chunk.doc_id}:{chunk.chunk_index}"))
    return chunk


def build_service(doc_path: Path, store: TrackingVectorStore, chunk: ChunkRecord) -> IngestService:
    settings = AppSettings()
    settings.ingest.atomic_alias_swap = True
    doc = DocumentText(doc_id="law-001", path=doc_path, pages=[chunk.body_text])
    reader_factory = DummyReaderFactory(doc)
    service = IngestService(
        settings,
        embedding_client=DummyEmbeddingClient(),
        vector_store=store,
        reader_factory=reader_factory,
    )
    service.verify_after_ingest = False
    service.builder.process_document = lambda *args, **kwargs: FakeProcessed(
        articles=[object()],
        chunks=[chunk],
        doc_ids_changed=[chunk.doc_id],
        doc_ids_unchanged=[],
    )
    return service


def test_reingest_same_document_keeps_point_count(tmp_path):
    doc_path = tmp_path / "law.txt"
    chunk = create_chunk()
    doc_path.write_text(chunk.body_text, encoding="utf-8")
    store = TrackingVectorStore()
    service = build_service(doc_path, store, chunk)

    service.ingest([doc_path])
    first_count = len(store.points)

    service.ingest([doc_path])
    second_count = len(store.points)

    assert first_count == second_count == 1
    assert store.swap_calls == [("alias", "alias_shadow"), ("alias", "alias_shadow")]
