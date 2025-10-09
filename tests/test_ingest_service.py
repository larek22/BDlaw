from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List

from app.data_repository import DataRepository
from app.ingest import IngestService
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

    def embed_texts(self, texts: Iterable[str], shas: Iterable[str], batch_size: int = 64, max_retries: int = 5) -> List[DummyEmbeddingResult]:
        vectors: List[DummyEmbeddingResult] = []
        for idx, sha in enumerate(shas):
            vectors.append(DummyEmbeddingResult(sha=sha, vector=[float(idx + 1)] * 3))
        return vectors


class DummyVectorStore:
    collection_name = "kb_docs_v1"

    def __init__(self) -> None:
        self.ensure_calls: List[tuple[str, bool]] = []
        self.deleted_ids: List[str] = []
        self.upserted: List[ChunkRecord] = []
        self.endpoint_url = "http://dummy"

    def ensure_collection(self, embedding_model: str, recreate: bool = False) -> None:
        self.ensure_calls.append((embedding_model, recreate))

    def delete_documents(self, doc_ids: Iterable[str]) -> None:
        self.deleted_ids.extend(doc_ids)

    def vector_size_for_model(self, model: str) -> int:
        return 3

    def count_points(self) -> int:
        return len(self.upserted)

    def upsert_chunks(self, chunks: Iterable[ChunkRecord], title_vectors, body_vectors) -> None:
        self.upserted = list(chunks)

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
                        "title_text": chunk.title_text,
                        "body_text": chunk.body_text,
                        "hierarchy": chunk.hierarchy,
                        "law_meta": chunk.law_meta,
                        "source": chunk.source,
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

    repository = DataRepository(tmp_path / "data")
    service = IngestService(
        settings,
        embedding_client=DummyEmbeddingClient(settings),
        vector_store=DummyVectorStore(),
        reader_factory=DummyReaderFactory(),
        repository=repository,
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
