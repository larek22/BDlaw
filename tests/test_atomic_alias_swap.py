from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from app.embeddings import EmbeddingResult
from app.ingest import IngestService
from app.legal_types import ChunkRecord
from app.settings import AppSettings


class StubEmbeddingClient:
    def reset_usage(self) -> None:
        pass

    def embed_texts(self, texts, shas, batch_size: int = 64, max_retries: int = 5):
        return [EmbeddingResult(sha=sha, vector=[1.0, 0.0, 0.5]) for sha in shas]

    def usage_summary(self) -> dict[str, float]:
        return {"requests": 0.0, "cache_hits": 0.0, "tokens": 0.0, "cost": 0.0}


class StubReaderFactory:
    def read(self, path: Path):
        return SimpleNamespace(full_text="Статья 1. Заголовок")


class StubBuilder:
    def __init__(self, chunk: ChunkRecord) -> None:
        self.chunk = chunk

    def process_document(self, document, normalized_text, chunk_size, overlap):
        return SimpleNamespace(
            articles=[],
            chunks=[self.chunk],
            doc_ids_changed=[self.chunk.doc_id],
            doc_ids_unchanged=[],
        )


class StubVectorStore:
    def __init__(self, healthy: bool) -> None:
        self.alias_name = "laws_active"
        self.collection_name = "laws_active"
        self.endpoint_url = "http://stub"
        self.healthy = healthy
        self.shadow_collection = "laws_active_shadow"
        self.shadow_count = 0
        self.active_count = 0
        self.swap_calls = 0
        self.cleanup_calls = 0
        self.deleted = False

    def ensure_collection(self, embedding_model: str, recreate: bool = False) -> None:
        pass

    def ensure_shadow_collection(self, alias: str, dim: int, on_disk: bool = True) -> str:
        self.shadow_collection = f"{alias}_shadow"
        return self.shadow_collection

    def vector_size_for_model(self, model: str) -> int:
        return 3

    def delete_documents(self, doc_ids):
        self.deleted = True

    def upsert_chunks(self, chunks, title_vectors, body_vectors, *, collection_override=None):
        self.shadow_count = len(chunks)
        if collection_override:
            self.shadow_collection = collection_override

    def count_points(self, collection: str | None = None) -> int:
        target = collection or self.collection_name
        if target == self.shadow_collection:
            return self.shadow_count
        return self.active_count

    def count_points_with_prefix(self, prefix: str) -> int:
        return self.shadow_count

    def count_points_for_doc_ids(self, doc_ids):
        return {doc_id: self.shadow_count for doc_id in doc_ids}

    def purge_orphans(self, prefixes):
        return 0

    def query(self, *args, **kwargs):
        return []

    def keyword_prefilter(self, *args, **kwargs):
        return []

    def build_doc_id_filter(self, doc_ids):
        return object()

    def combine_filters(self, *filters):
        return None

    def health_probe(self, collection: str, sample_texts, limit: int = 5) -> bool:
        return self.healthy

    def swap_alias_atomically(self, alias: str, new_collection: str) -> None:
        self.swap_calls += 1
        self.active_count = self.shadow_count

    def cleanup_old_collections(self, alias: str, keep_n: int = 2) -> None:
        self.cleanup_calls += 1


def _make_chunk() -> ChunkRecord:
    return ChunkRecord(
        doc_id="law:v1:art001",
        chunk_index=0,
        title_text="Статья 1",
        body_text="Пример правового текста",
        hierarchy={"article_no": "1", "article_no_int": 1},
        law_meta={"status": "in_force"},
        chunk_sha256="chunk-0",
        title_sha256="title-0",
        body_sha256="body-0",
    )


def _make_service(settings: AppSettings, vector_store: StubVectorStore) -> IngestService:
    chunk = _make_chunk()
    service = IngestService(
        settings,
        embedding_client=StubEmbeddingClient(),
        vector_store=vector_store,
        reader_factory=StubReaderFactory(),
        repository=None,
        verify_after_ingest=False,
    )
    service.builder = StubBuilder(chunk)
    return service


def test_alias_not_swapped_when_health_probe_fails():
    settings = AppSettings()
    settings.ingest.atomic_alias_swap = True
    settings.ingest.validation_sample_k = 2
    vector_store = StubVectorStore(healthy=False)
    service = _make_service(settings, vector_store)

    stats = service.ingest([Path("/tmp/fake.pdf")])

    assert stats.chunks_created == 1
    assert vector_store.swap_calls == 0
    assert vector_store.cleanup_calls == 0
    assert vector_store.deleted is False


def test_alias_swapped_after_successful_validation():
    settings = AppSettings()
    settings.ingest.atomic_alias_swap = True
    settings.ingest.validation_sample_k = 2
    vector_store = StubVectorStore(healthy=True)
    service = _make_service(settings, vector_store)

    stats = service.ingest([Path("/tmp/fake.pdf")])

    assert stats.chunks_created == 1
    assert vector_store.swap_calls == 1
    assert vector_store.cleanup_calls == 1
    assert vector_store.count_points(vector_store.shadow_collection) == 1
    assert vector_store.active_count == 1
