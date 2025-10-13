from __future__ import annotations

from pathlib import Path
import uuid

from app.data_repository import DataRepository
from app.legal_pipeline import LegalCorpusBuilder, normalize_text
from app.readers.base import DocumentText


def test_normalize_text_removes_extra_whitespace() -> None:
    raw = "Line 1\r\nLine-\n 2\n\n\nLine   3"
    cleaned = normalize_text(raw)
    assert "Line-" not in cleaned
    assert "Line 3" in cleaned
    assert "\n\n\n" not in cleaned


def test_legal_corpus_builder_creates_articles(tmp_path: Path) -> None:
    repository = DataRepository(tmp_path / "data")
    builder = LegalCorpusBuilder(repository)

    text = """
    РАЗДЕЛ I. ТЕСТОВЫЙ РАЗДЕЛ
    ГЛАВА 2. ПРОВЕРКА
    СТАТЬЯ 10. Тестовая статья
    Текст первого абзаца.
    Текст второго абзаца.
    """.strip()

    document = DocumentText(
        doc_id="test.rtf",
        path=repository.paths.raw / "gk_rf" / "part_1" / "test.rtf",
        pages=[text],
    )

    processed = builder.process_document(
        document,
        normalized_text=normalize_text(text),
        chunk_size=120,
        overlap=20,
    )

    assert processed.articles
    assert processed.chunks
    first_chunk = processed.chunks[0]
    assert first_chunk.doc_id.startswith("gkrf:part")
    assert first_chunk.hierarchy["article_no"] == "10"
    # Chunk identifiers must be UUID strings so they are valid Qdrant point IDs.
    uuid_obj = uuid.UUID(first_chunk.chunk_id)
    assert str(uuid_obj) == first_chunk.chunk_id


def test_builder_detects_case_insensitive_headings(tmp_path: Path) -> None:
    repository = DataRepository(tmp_path / "data")
    builder = LegalCorpusBuilder(repository)

    text = """
    раздел iv. тестовый раздел
    глава 73. проверка кейса
    статья 1335.1. ещё одна проверка
    Первый абзац статьи.
    Второй абзац статьи.
    """.strip()

    document = DocumentText(
        doc_id="lower.rtf",
        path=repository.paths.raw / "gk_rf" / "part_4" / "lower.rtf",
        pages=[text],
    )

    processed = builder.process_document(
        document,
        normalized_text=normalize_text(text),
        chunk_size=120,
        overlap=20,
    )

    assert processed.articles, "Expected at least one parsed article"
    first_article = processed.articles[0]
    assert first_article.hierarchy.chapter_no == 73
    assert first_article.hierarchy.article_no == "1335.1"
    assert any(
        chunk.hierarchy.get("chapter_no") == 73 for chunk in processed.chunks
    ), "Chunks should inherit detected chapter metadata"
