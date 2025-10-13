from __future__ import annotations

from pathlib import Path

from app.ingest import IngestService
from app.legal_types import ChunkRecord
from app.readers.base import DocumentText
from app.legal_pipeline import ProcessedDocument
from app.settings import AppSettings
from app.data_repository import DataRepository

from app.embeddings import EmbeddingResult


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

    def vector_size_for_model(self, model: str) -> int:
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

    def reset_usage(self) -> None:
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


def test_idempotent_upsert(tmp_path):
    settings = AppSettings()
    settings.ingest.atomic_alias_swap = True
    settings.ingest.enable_new_loaders = False
    settings.ingest.dry_run = False
    settings.qdrant.prefer_grpc = False
    settings.qdrant.url = "http://localhost:6333"
    settings.qdrant.api_key = None
    repository = DataRepository(root=tmp_path / "repo")
    chunk = ChunkRecord(
        doc_id="doc-1",
        chunk_index=0,
        title_text="Title",
        body_text="Body",
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

    stats_first = service._ingest_atomic([Path("doc1.txt")])
    stats_second = service._ingest_atomic([Path("doc1.txt")])

    assert stats_first.chunks_created == stats_second.chunks_created == 1
    assert all(len(points) == 1 for points in vector_store.points_by_collection.values())
