from __future__ import annotations

from pathlib import Path

import pytest

from app.embeddings import EmbeddingResult
from app.ingest import IngestService
from app.legal_types import ChunkRecord
from app.readers.base import DocumentText
from app.legal_pipeline import ProcessedDocument
from app.settings import AppSettings
from app.data_repository import DataRepository


class RecordingVectorStore:
    alias_name = "kb_docs_active"
    collection_name = "kb_docs_active"
    endpoint_url = "http://test"

    def __init__(self) -> None:
        self.points_by_collection: dict[str, set[str]] = {}
        self.collection_counter = 0
        self.swapped_to: str | None = None
        self.cleanup_called = False
        self.force_bad_count = False
        self.health = True

    def vector_size_for_model(self, model: str) -> int:  # pragma: no cover - simple
        return 3

    def ensure_shadow_collection(self, alias: str, dim: int, on_disk: bool = True) -> str:
        self.collection_counter += 1
        target = f"{alias}_shadow_{self.collection_counter}"
        self.points_by_collection.setdefault(target, set())
        return target

    def upsert_chunks(self, chunks, title_vectors, body_vectors, *, collection_override=None):
        collection = collection_override or self.collection_name
        store = self.points_by_collection.setdefault(collection, set())
        for chunk in chunks:
            store.add(chunk.chunk_id)

    def count_points(self, collection: str | None = None) -> int:
        if self.force_bad_count:
            return 0
        target = collection or self.collection_name
        return len(self.points_by_collection.get(target, set()))

    def health_probe(self, collection: str, sample_texts, limit: int = 5) -> bool:
        return self.health

    def swap_alias_atomically(self, alias: str, new_collection: str) -> None:
        self.swapped_to = new_collection

    def cleanup_old_collections(self, alias: str, keep_n: int = 2) -> None:
        self.cleanup_called = True


class StubEmbeddingClient:
    def __init__(self) -> None:
        self.vector = [0.1, 0.2, 0.3]

    def reset_usage(self) -> None:  # pragma: no cover - no-op
        return None

    def embed_texts(self, texts, shas, batch_size: int = 64, max_retries: int = 5):
        return [EmbeddingResult(sha=sha, vector=self.vector) for sha in shas]


class StubReaderFactory:
    def __init__(self, document: DocumentText) -> None:
        self._document = document

    def read(self, path: Path) -> DocumentText:
        return self._document


class StubBuilder:
    def __init__(self, repository: DataRepository, chunk: ChunkRecord) -> None:
        self.repository = repository
        self._chunk = chunk

    def process_document(self, document, *, normalized_text, chunk_size, overlap):
        return ProcessedDocument(
            normalized_text_path=self.repository.paths.staging / "norm.txt",
            article_json_path=self.repository.paths.staging / "articles.jsonl",
            chunk_json_path=self.repository.paths.staging / "chunks.jsonl",
            articles=[],
            chunks=[self._chunk],
            doc_ids_changed=[self._chunk.doc_id],
            doc_ids_unchanged=[],
        )


@pytest.fixture()
def sample_components(tmp_path):
    settings = AppSettings()
    settings.ingest.atomic_alias_swap = True
    settings.ingest.enable_new_loaders = False
    settings.ingest.validation_sample_k = 3
    settings.ingest.dry_run = False
    settings.qdrant.prefer_grpc = False
    settings.qdrant.url = "http://localhost:6333"
    settings.qdrant.api_key = None
    repository = DataRepository(root=tmp_path / "repo")
    chunk = ChunkRecord(
        doc_id="doc-1",
        chunk_index=0,
        title_text="Заголовок",
        body_text="Статья 1. Тест",
        chunk_sha256="chunk-sha",
        title_sha256="title-sha",
        body_sha256="body-sha",
    )
    document = DocumentText(doc_id="doc-1", path=tmp_path / "doc1.txt", pages=["Статья 1"])
    reader_factory = StubReaderFactory(document)
    builder = StubBuilder(repository, chunk)
    vector_store = RecordingVectorStore()
    embedding_client = StubEmbeddingClient()
    service = IngestService(
        settings,
        embedding_client,
        vector_store,
        reader_factory=reader_factory,
        repository=repository,
    )
    service.builder = builder
    return service, vector_store, settings


def test_atomic_alias_swap_success(sample_components):
    service, vector_store, _ = sample_components
    stats = service._ingest_atomic([Path("doc1.txt")])
    assert stats.chunks_created == 1
    assert vector_store.swapped_to is not None
    assert vector_store.cleanup_called is True


def test_atomic_alias_swap_validation_failure(sample_components):
    service, vector_store, _ = sample_components
    vector_store.force_bad_count = True
    with pytest.raises(RuntimeError):
        service._ingest_atomic([Path("doc1.txt")])
    assert vector_store.swapped_to is None
