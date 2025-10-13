from __future__ import annotations

from pathlib import Path
from typing import Callable

from app.ingest import IngestService
from app.legal_pipeline import ProcessedDocument
from app.legal_types import ChunkRecord
from app.settings import AppSettings


class FakeEmbeddingClient:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def embed_texts(self, texts, shas, batch_size: int = 64, max_retries: int = 5):
        from app.embeddings import EmbeddingResult

        return [EmbeddingResult(sha=sha, vector=[0.1, 0.2, 0.3, 0.4]) for sha in shas]

    def reset_usage(self) -> None:  # pragma: no cover - interface stub
        return None

    def usage_summary(self) -> dict[str, float]:  # pragma: no cover - interface stub
        return {"cache_hits": 0.0, "requests": 0.0, "tokens": 0.0, "cost": 0.0}


class FakeVectorStore:
    alias_name = "alias"
    collection_name = "alias"
    endpoint_url = "http://localhost:6333"

    def __init__(self, expected_dim: int = 4) -> None:
        self.expected_dim = expected_dim
        self.shadow_collection = "alias_shadow"
        self.health_ok = True
        self.count_override: int | None = None
        self.swapped = False
        self.cleaned = False
        self.upsert_calls: list[tuple[str, int]] = []
        self.points: dict[str, dict[str, set[str]]] = {}
        self.last_upsert_chunks: list[ChunkRecord] = []

    def ensure_collection(self, embedding_model: str, recreate: bool = False) -> None:
        return None

    def vector_size_for_model(self, model: str) -> int:
        return self.expected_dim

    def ensure_shadow_collection(self, alias: str, dim: int, on_disk: bool = True) -> str:
        self.shadow_collection = f"{alias}_shadow"
        return self.shadow_collection

    def upsert_chunks(self, chunks, title_vectors, body_vectors, *, collection_name: str | None = None) -> None:
        target = collection_name or self.collection_name
        chunk_list = list(chunks)
        self.upsert_calls.append((target, len(chunk_list)))
        self.last_upsert_chunks = chunk_list
        bucket = self.points.setdefault(target, {})
        for chunk in chunk_list:
            bucket.setdefault(chunk.doc_id, set()).add(chunk.chunk_id)

    def count_points(self, collection: str | None = None) -> int:
        if self.count_override is not None:
            return self.count_override
        target = collection or self.collection_name
        bucket = self.points.get(target, {})
        return sum(len(ids) for ids in bucket.values())

    def count_points_for_doc_ids(self, collection: str, doc_ids) -> dict[str, int]:
        bucket = self.points.get(collection or self.collection_name, {})
        return {doc_id: len(bucket.get(doc_id, set())) for doc_id in doc_ids}

    def health_probe(self, collection: str, sample_texts: list[str], limit: int = 5) -> bool:
        return self.health_ok

    def swap_alias_atomically(self, alias: str, new_collection: str) -> None:
        self.swapped = True
        self.last_swap = (alias, new_collection)

    def cleanup_old_collections(self, alias: str, keep_n: int = 2) -> None:
        self.cleaned = True


def make_chunk(doc_id: str, index: int) -> ChunkRecord:
    chunk = ChunkRecord(
        doc_id=doc_id,
        chunk_index=index,
        title_text=f"Статья {index+1}",
        body_text="Текст",
    )
    chunk.title_sha256 = f"title-{index}"
    chunk.body_sha256 = f"body-{index}"
    chunk.chunk_sha256 = f"chunk-{index}"
    return chunk


def make_processed() -> ProcessedDocument:
    return ProcessedDocument(
        article_json_path=Path("articles.json"),
        chunk_json_path=Path("chunks.json"),
        normalized_text_path=Path("normalized.txt"),
        manifest_path=Path("manifest.json"),
        pending_manifest_path=Path("manifest.json.pending"),
        articles=[],
        chunks=[],
        doc_ids_changed=[],
        doc_ids_unchanged=[],
    )


def build_service(settings: AppSettings, vector_store: FakeVectorStore) -> IngestService:
    embedding_client = FakeEmbeddingClient(settings)
    
    class _StubFactory:
        def __init__(self) -> None:
            self.settings = settings

        def read(self, path: Path):  # pragma: no cover - not exercised in unit tests
            raise NotImplementedError(path)

    service = IngestService(
        settings=settings,
        embedding_client=embedding_client,
        vector_store=vector_store,
        repository=None,
        reader_factory=_StubFactory(),
        verify_after_ingest=False,
    )
    return service


def run_atomic(
    service: IngestService,
    vector_store: FakeVectorStore,
    *,
    count_override: int | None,
    health_ok: bool,
    report: Callable[..., None] | None = None,
):
    vector_store.count_override = count_override
    vector_store.health_ok = health_ok
    chunk = make_chunk("doc-1", 0)
    return service._ingest_atomic_flow(
        report=report or (lambda *_args, **_kwargs: None),
        processed_documents=[make_processed()],
        changed_chunks=[chunk],
        title_vectors={chunk.title_sha256: [0.1, 0.2, 0.3, 0.4]},
        body_vectors={chunk.body_sha256: [0.1, 0.2, 0.3, 0.4]},
        embedding_model=service.settings.openai_models.embedding,
        skipped=0,
        doc_expected_counts={chunk.doc_id: 1},
    )


__all__ = [
    "FakeEmbeddingClient",
    "FakeVectorStore",
    "make_chunk",
    "make_processed",
    "build_service",
    "run_atomic",
]
