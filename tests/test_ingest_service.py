from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List

from tests_helpers_ingest import (
    FakeVectorStore,
    build_service,
    make_chunk,
    make_processed,
)

from app.data_repository import DataRepository
from app.ingest import IngestService, make_point_id
from app.legal_pipeline import ProcessedDocument
from app.legal_types import ChunkRecord
from app.readers.base import DocumentText
from app.settings import AppSettings


_ARTICLE_TEXT = """
РАЗДЕЛ I. ОБЩИЕ ПОЛОЖЕНИЯ
ГЛАВА 1. ОСНОВЫ
СТАТЬЯ 1. Общая статья
Первый абзац.
Второй абзац.
""".strip()


class DummyReaderFactory:
    def read(self, path: Path) -> DocumentText:
        return DocumentText(doc_id=path.name, path=path, pages=[_ARTICLE_TEXT])


@dataclass
class DummyEmbeddingResult:
    sha: str
    vector: List[float]


class DummyEmbeddingClient:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings
        self.reset_usage()

    def embed_texts(self, texts: Iterable[str], shas: Iterable[str], batch_size: int = 64, max_retries: int = 5) -> List[DummyEmbeddingResult]:
        vectors: List[DummyEmbeddingResult] = []
        for idx, sha in enumerate(shas):
            vectors.append(DummyEmbeddingResult(sha=sha, vector=[float(idx + 1)] * 3))
        self._requests += 1
        self._tokens += sum(len(text) for text in texts)
        return vectors

    def reset_usage(self) -> None:
        self._requests = 0
        self._cached = 0
        self._tokens = 0

    def usage_summary(self) -> dict[str, float]:
        return {
            "requests": float(self._requests),
            "cache_hits": float(self._cached),
            "tokens": float(self._tokens),
            "cost": 0.0,
        }


class DummyVectorStore:
    collection_name = "kb_docs_v1"

    def __init__(self) -> None:
        self.ensure_calls: List[tuple[str, bool]] = []
        self.deleted_ids: List[str] = []
        self.upserted: List[ChunkRecord] = []
        self.endpoint_url = "http://dummy"
        self.purged_prefixes: List[str] = []

    def ensure_collection(self, embedding_model: str, recreate: bool = False) -> None:
        self.ensure_calls.append((embedding_model, recreate))

    def delete_documents(self, doc_ids: Iterable[str]) -> None:
        self.deleted_ids.extend(doc_ids)

    def vector_size_for_model(self, model: str) -> int:
        return 3

    def count_points(self, collection: str | None = None) -> int:
        return len(self.upserted)

    def count_points_with_prefix(self, prefix: str) -> int:
        return sum(1 for chunk in self.upserted if chunk.doc_id.startswith(prefix))

    def count_points_for_doc_ids(self, collection: str, doc_ids: Iterable[str]) -> Dict[str, int]:
        counts: Dict[str, int] = {doc_id: 0 for doc_id in doc_ids}
        for chunk in self.upserted:
            if chunk.doc_id in counts:
                counts[chunk.doc_id] += 1
        return counts

    def upsert_chunks(self, chunks: Iterable[ChunkRecord], title_vectors, body_vectors) -> None:
        self.upserted = list(chunks)

    def purge_orphans(self, prefixes: Iterable[str]) -> int:
        self.purged_prefixes = list(prefixes)
        return 0

    def query(self, *, vector_name: str, query_vector: List[float], limit: int, filters=None) -> List[SimpleNamespace]:
        results: List[SimpleNamespace] = []
        if self.upserted:
            chunk = self.upserted[0]
            results.append(
                SimpleNamespace(
                    id=chunk.chunk_id,
                    score=1.0,
                    payload={
                        "doc_id": chunk.doc_id,
                        "chunk_index": chunk.chunk_index,
                        "chunk_id": chunk.chunk_id,
                        "chunk_key": chunk.chunk_key,
                        "title_text": chunk.title_text,
                        "body_text": chunk.body_text,
                        "hierarchy": chunk.hierarchy,
                        "law_meta": chunk.law_meta,
                        "source": chunk.source,
                        "plan_version": chunk.plan_version,
                        "parser_version": chunk.parser_version,
                        "chunk_sha256": chunk.chunk_sha256,
                        "title_sha256": chunk.title_sha256,
                        "body_sha256": chunk.body_sha256,
                    },
                )
            )
        return results[:limit]

    def keyword_prefilter(self, query: str, limit: int) -> List[str]:  # pragma: no cover - not used
        return []

    def build_doc_id_filter(self, doc_ids: Iterable[str]):  # pragma: no cover - not used
        return list(doc_ids)


def test_ingest_writes_chunks_and_logs_progress(tmp_path: Path) -> None:
    settings = AppSettings()
    settings.openai_models.embedding = "dummy-embed"
    settings.ingest.atomic_alias_swap = False

    repository = DataRepository(tmp_path / "data")
    service = IngestService(
        settings,
        embedding_client=DummyEmbeddingClient(settings),
        vector_store=DummyVectorStore(),
        reader_factory=DummyReaderFactory(),
        repository=repository,
        verify_after_ingest=False,
    )
    service.builder.tokenizer = SimpleNamespace(
        encode=lambda text, disallowed_special=(): list(text)
    )

    raw_file = repository.paths.raw / "gk_rf" / "part_1" / "law.rtf"
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    raw_file.touch()

    progress: List[str] = []
    stats = service.ingest([raw_file], progress_cb=progress.append)

    assert stats.files_processed == 1
    assert stats.chunks_created >= 1
    assert any("Embedding" in message for message in progress)
    assert any("Sample chunk" in message for message in progress)
    assert any("total points now" in message for message in progress)
    assert any("[VERIFY]" in message for message in progress)


def test_force_reingest_when_collection_empty(tmp_path: Path) -> None:
    settings = AppSettings()
    settings.ingest.atomic_alias_swap = False
    settings.ingest.force_reingest_if_empty = True

    repository = DataRepository(tmp_path / "data")

    class ForceVectorStore(DummyVectorStore):
        def __init__(self) -> None:
            super().__init__()
            self._count_calls = 0

        def count_points(self, collection: str | None = None) -> int:
            self._count_calls += 1
            if self._count_calls == 1:
                return 0
            return len(self.upserted)

    vector_store = ForceVectorStore()

    service = IngestService(
        settings,
        embedding_client=DummyEmbeddingClient(settings),
        vector_store=vector_store,
        reader_factory=DummyReaderFactory(),
        repository=repository,
        verify_after_ingest=False,
    )
    service.builder.tokenizer = SimpleNamespace(
        encode=lambda text, disallowed_special=(): list(text)
    )

    def fake_process(document, normalized_text: str, chunk_size: int, overlap: int):
        doc_id = "gkrf:part1:art1:v2024-01-01"
        chunk = ChunkRecord(doc_id=doc_id, chunk_index=0, title_text="T", body_text="B")
        chunk.title_sha256 = "title-sha"
        chunk.body_sha256 = "body-sha"
        chunk.chunk_sha256 = "chunk-sha"
        chunk.chunk_id = make_point_id(chunk.doc_id, chunk.chunk_index)

        manifest_dir = tmp_path / "staging"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / "doc.manifest.json"
        pending_manifest_path = manifest_dir / "doc.manifest.json.pending"
        pending_manifest_path.write_text("{}", encoding="utf-8")

        return ProcessedDocument(
            normalized_text_path=manifest_dir / "normalized.txt",
            article_json_path=manifest_dir / "articles.json",
            chunk_json_path=manifest_dir / "chunks.json",
            manifest_path=manifest_path,
            pending_manifest_path=pending_manifest_path,
            articles=[],
            chunks=[chunk],
            doc_ids_changed=[],
            doc_ids_unchanged=[doc_id],
        )

    service.builder.process_document = fake_process  # type: ignore[assignment]

    raw_file = repository.paths.raw / "gk" / "part_1" / "law.rtf"
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    raw_file.write_text("stub", encoding="utf-8")

    progress: List[str] = []
    stats = service.ingest([raw_file], progress_cb=progress.append)

    assert stats.chunks_created == 1
    assert vector_store.upserted
    assert any("forcing re-ingest" in message.lower() for message in progress)


def test_atomic_flow_waits_for_async_counts(tmp_path: Path) -> None:
    settings = AppSettings()
    settings.ingest.atomic_alias_swap = True
    settings.ingest.dry_run = True
    settings.qdrant.use_wait = False

    repository = DataRepository(tmp_path / "data")

    class WaitVectorStore(DummyVectorStore):
        def __init__(self) -> None:
            super().__init__()
            self.wait_calls: List[tuple[str, int]] = []
            self._shadow_name = "alias_active_d3_shadow"

        @property
        def alias_name(self) -> str:
            return "alias_active"

        def ensure_collection(self, embedding_model: str, recreate: bool = False) -> None:
            self.ensure_calls.append((embedding_model, recreate))

        def ensure_shadow_collection(self, alias: str, dim: int, on_disk: bool = True) -> str:
            assert alias == self.alias_name
            assert dim == self.vector_size_for_model("")
            return self._shadow_name

        def upsert_chunks(self, chunks, title_vectors, body_vectors, collection_name=None):
            self.upserted = list(chunks)

        def wait_for_count(self, collection: str, expected: int) -> int:
            self.wait_calls.append((collection, expected))
            return expected

        def health_probe(self, collection: str, sample_texts: List[str], limit: int = 5) -> bool:
            return True

        def count_points(self, collection: str | None = None) -> int:  # pragma: no cover - fallback
            return len(self.upserted)

    vector_store = WaitVectorStore()

    service = IngestService(
        settings,
        embedding_client=DummyEmbeddingClient(settings),
        vector_store=vector_store,
        reader_factory=DummyReaderFactory(),
        repository=repository,
        verify_after_ingest=False,
    )

    def fake_process(document, normalized_text: str, chunk_size: int, overlap: int):
        doc_id = "gkrf:part1:art1:v2024-01-01"
        chunk = ChunkRecord(doc_id=doc_id, chunk_index=0, title_text="T", body_text="B")
        chunk.title_sha256 = "title-sha"
        chunk.body_sha256 = "body-sha"
        chunk.chunk_sha256 = "chunk-sha"
        chunk.chunk_id = make_point_id(chunk.doc_id, chunk.chunk_index)

        staging_root = tmp_path / "staging"
        staging_root.mkdir(parents=True, exist_ok=True)
        manifest_path = staging_root / "doc.manifest.json"
        pending_manifest_path = staging_root / "doc.manifest.json.pending"
        pending_manifest_path.write_text("{}", encoding="utf-8")

        return ProcessedDocument(
            normalized_text_path=staging_root / "normalized.txt",
            article_json_path=staging_root / "articles.json",
            chunk_json_path=staging_root / "chunks.json",
            manifest_path=manifest_path,
            pending_manifest_path=pending_manifest_path,
            articles=[],
            chunks=[chunk],
            doc_ids_changed=[doc_id],
            doc_ids_unchanged=[],
        )

    service.builder.process_document = fake_process  # type: ignore[assignment]

    raw_file = repository.paths.raw / "gk" / "part_1" / "law.rtf"
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    raw_file.write_text("stub", encoding="utf-8")

    stats = service.ingest([raw_file])

    assert stats.chunks_created == 1
    assert vector_store.wait_calls
    assert vector_store.wait_calls[0][0] == vector_store._shadow_name
    assert vector_store.wait_calls[0][1] == 1
    assert stats.alias_swapped is False
    assert stats.validation_passed is True


def test_atomic_flow_deduplicates_duplicate_chunk_ids(tmp_path: Path) -> None:
    settings = AppSettings()
    settings.ingest.atomic_alias_swap = True
    settings.ingest.dry_run = True

    vector_store = FakeVectorStore(expected_dim=4)
    service = build_service(settings, vector_store)

    primary = make_chunk("doc-1", 0)
    duplicate_identical = replace(primary)
    conflicting = replace(primary)
    conflicting.body_text = "Обновленный текст"
    conflicting.body_sha256 = "body-updated"
    conflicting.chunk_sha256 = "chunk-updated"

    title_vectors = {
        primary.title_sha256: [0.1, 0.2, 0.3, 0.4],
    }
    body_vectors = {
        primary.body_sha256: [0.4, 0.3, 0.2, 0.1],
        conflicting.body_sha256: [0.9, 0.8, 0.7, 0.6],
    }

    stats = service._ingest_atomic_flow(
        report=lambda *_args, **_kwargs: None,
        processed_documents=[make_processed()],
        changed_chunks=[primary, duplicate_identical, conflicting],
        title_vectors=title_vectors,
        body_vectors=body_vectors,
        embedding_model=settings.openai_models.embedding,
        skipped=0,
        doc_expected_counts={primary.doc_id: 3},
    )

    assert stats.chunks_created == 1
    assert vector_store.upsert_calls[-1][1] == 1
    assert vector_store.last_upsert_chunks[0].body_text == conflicting.body_text
