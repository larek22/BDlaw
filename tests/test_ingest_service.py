from pathlib import Path
from types import SimpleNamespace

from app.ingest import IngestService
from app.readers.base import DocumentText
from app.settings import AppSettings


class DummyReaderFactory:
    def read(self, path: Path) -> DocumentText:
        return DocumentText(doc_id=path.name, path=path, pages=["This is a test document."])


class DummyEmbeddingClient:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def embed_texts(self, texts, shas, batch_size: int = 64, max_retries: int = 5):
        return [DummyEmbeddingResult(sha=sha, vector=[float(index + 1)] * 3) for index, sha in enumerate(shas)]


class DummyEmbeddingResult:
    def __init__(self, sha: str, vector: list[float]) -> None:
        self.sha = sha
        self.vector = vector


class DummyVectorStore:
    collection_name = "kb_docs_v1"

    def __init__(self) -> None:
        self.points: list[tuple[object, list[float]]] = []
        self.ensure_calls: list[tuple[str, bool]] = []
        self.endpoint_url = "http://test-qdrant"

    def ensure_collection(self, *, embedding_model: str, recreate: bool = False) -> None:
        self.ensure_calls.append((embedding_model, recreate))

    def vector_size_for_model(self, model: str) -> int:
        return 3

    def count_points(self) -> int:
        return len(self.points)

    def upsert_chunks(self, chunks, vectors, **kwargs) -> None:  # pragma: no cover - simple stub
        self.points = list(zip(list(chunks), list(vectors)))

    def search(self, query_vector, top_k: int):
        if not self.points:
            return []
        chunk, _ = self.points[0]
        return [SimpleNamespace(payload={"doc_id": chunk.doc_id, "chunk_index": chunk.chunk_index}, score=0.5)]

    def point_id_for_chunk(self, chunk, **kwargs):  # pragma: no cover - deterministic stub
        return f"point-{chunk.chunk_index}"


def test_ingest_emits_progress_and_verification_messages(tmp_path):
    settings = AppSettings()
    settings.openai_models.embedding = "stub-model"

    service = IngestService(
        settings,
        embedding_client=DummyEmbeddingClient(settings),
        vector_store=DummyVectorStore(),
        reader_factory=DummyReaderFactory(),
    )

    progress: list[str] = []
    stats = service.ingest([tmp_path / "doc1.txt"], progress_cb=progress.append)

    assert stats.files_processed == 1
    assert any("Embedding" in message for message in progress)
    assert any("Sample chunk" in message for message in progress)
    assert any("Sample point id" in message for message in progress)
    assert any("total points now" in message for message in progress)
    assert any("[VERIFY]" in message for message in progress)
