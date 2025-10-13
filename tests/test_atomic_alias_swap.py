from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pytest

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


class DummyVectorStore:
    collection_name = "alias"
    endpoint_url = "http://example"

    def __init__(self, count_result: int, health_ok: bool) -> None:
        self.alias_name = "alias"
        self.count_map = {"alias_shadow": count_result}
        self.health_ok = health_ok
        self.swap_calls: List[tuple[str, str]] = []
        self.cleaned = False
        self.last_collection = None

    def vector_size_for_model(self, model: str) -> int:
        return 3

    def ensure_shadow_collection(self, alias: str, dim: int) -> str:
        self.last_collection = f"{alias}_shadow"
        return self.last_collection

    def upsert_chunks(self, chunks, title_vectors, body_vectors, *, collection: str | None = None):
        # emulate count refresh for validation
        unique_ids = {chunk.chunk_id for chunk in chunks}
        if collection:
            self.count_map[collection] = len(unique_ids)

    def count_points(self, collection: str | None = None) -> int:
        return self.count_map.get(collection or self.collection_name, 0)

    def health_probe(self, collection: str, sample_texts: list[str], limit: int = 5) -> bool:
        return self.health_ok

    def swap_alias_atomically(self, alias: str, new_collection: str) -> None:
        self.swap_calls.append((alias, new_collection))

    def cleanup_old_collections(self, alias: str, keep_n: int = 2) -> None:
        self.cleaned = True

    def keyword_prefilter(self, query: str, limit: int = 0):  # pragma: no cover - unused in tests
        return []

    def build_doc_id_filter(self, doc_ids):  # pragma: no cover - unused in tests
        raise NotImplementedError

    def build_as_of_filter(self, **kwargs):  # pragma: no cover - unused
        return None

    def combine_filters(self, *filters):  # pragma: no cover - unused
        return None


@pytest.fixture
def sample_chunk() -> ChunkRecord:
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


def build_service(doc_path: Path, vector_store, chunk: ChunkRecord) -> IngestService:
    settings = AppSettings()
    settings.ingest.atomic_alias_swap = True
    settings.ingest.validation_sample_k = 2
    settings.ingest.dry_run = False
    doc = DocumentText(doc_id="law-001", path=doc_path, pages=[chunk.body_text])
    reader_factory = DummyReaderFactory(doc)
    service = IngestService(
        settings,
        embedding_client=DummyEmbeddingClient(),
        vector_store=vector_store,
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


def test_atomic_swap_success(tmp_path, sample_chunk):
    doc_path = tmp_path / "law.txt"
    doc_path.write_text(sample_chunk.body_text, encoding="utf-8")
    store = DummyVectorStore(count_result=1, health_ok=True)
    service = build_service(doc_path, store, sample_chunk)
    stats = service.ingest([doc_path])
    assert stats.chunks_created == 1
    assert store.swap_calls == [("alias", "alias_shadow")]
    assert store.cleaned is True


def test_atomic_swap_aborts_on_validation(tmp_path, sample_chunk):
    doc_path = tmp_path / "law.txt"
    doc_path.write_text(sample_chunk.body_text, encoding="utf-8")
    store = DummyVectorStore(count_result=0, health_ok=False)
    service = build_service(doc_path, store, sample_chunk)
    stats = service.ingest([doc_path])
    assert stats.chunks_created == 1
    assert store.swap_calls == []
    assert store.cleaned is False
