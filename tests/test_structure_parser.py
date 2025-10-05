from datetime import date
from pathlib import Path

from bdlaw.index.ids import make_hash
from bdlaw.ingest.base import Document
from bdlaw.normalize import normalize_document
from bdlaw.parser.structure import ParsingContext, parse_document


def build_document(text: str) -> Document:
    return Document(
        path=Path("law.txt"),
        mime_type="text/plain",
        raw_text=text,
        metadata={"clean_text": text, "page_spans": [[1, 1]]},
    )


def build_context(version: str = "2024-01-01") -> ParsingContext:
    return ParsingContext(
        jurisdiction="ru",
        act_type="federal_law",
        law_code="ГК РФ",
        law_full_title="Гражданский кодекс РФ",
        version_id=version,
        valid_from=date.fromisoformat(version),
    )


def test_parse_simple_article_with_point():
    text = """
    Статья 16.1. Заголовок
    Часть 1. Пункт 1. Текст пункта.
    Часть 2. Дополнительный текст.
    """
    doc = build_document(text)
    parsed = parse_document(doc, build_context())
    assert len(parsed.norms) == 2
    first = parsed.norms[0]
    assert first.article == "16.1"
    assert first.part == "1"
    assert first.point == "1"
    assert first.clean_text.startswith("Текст пункта")
    second = parsed.norms[1]
    assert second.part == "2"
    assert second.point is None


def test_semicolon_lists_split_into_subpoints():
    text = """
    Статья 12. Часть 1. Гражданские права могут защищаться:
    пункт 1. возмещением убытков; взысканием неустойки; иными способами.
    """
    parsed = parse_document(build_document(text), build_context())
    # базовый пункт + три подпункта
    assert len(parsed.norms) == 4
    base = parsed.norms[0]
    assert base.subpoint is None
    subpoints = [norm for norm in parsed.norms[1:]]
    assert {norm.subpoint for norm in subpoints} == {"1", "2", "3"}
    assert all(norm.point == "1" for norm in subpoints)
    assert parsed.structure.articles


def test_amendments_annotations_and_cross_refs_extracted():
    text = (
        "Статья 395. Проценты (в ред. Федерального закона от 30.12.2012 № 302-ФЗ). "
        "Применяется согласно ст. 12 ГК РФ. (Постановление Конституционного суда РФ от 03.07.2019 № 26-П)."
    )
    parsed = parse_document(build_document(text), build_context())
    norm = parsed.norms[0]
    assert norm.amendments == [{"date": "2012-12-30", "law_no": "302-ФЗ", "scope": "article"}]
    assert norm.annotations[0]["type"] == "cc_ruling"
    assert norm.cross_refs[0]["article"] == "12"
    assert norm.hash == make_hash(norm.clean_text)


def test_citation_format_includes_version():
    parsed = parse_document(build_document("Статья 12. Часть 1. Текст."), build_context("2024-02-02"))
    assert "2024-02-02" in parsed.norms[0].citation


def test_span_offsets_align_with_raw_text():
    raw_text = "Статья 5.\nПункт 1. Сло-\nво восстановлено."
    document = normalize_document(Document(path=Path("law.txt"), mime_type="text/plain", raw_text=raw_text, metadata={}))
    document.metadata.setdefault("page_spans", [[1, 1]])
    parsed = parse_document(document, build_context())
    norm = parsed.norms[0]
    start, end = norm.span_offsets[0]
    assert document.raw_text[start:end] == norm.raw_text


def test_subpoint_offsets_use_raw_coordinates():
    text = (
        "Статья 12. Пункт 1. Возмещением убытков; взысканием неустойки; иными способами."
    )
    document = normalize_document(Document(path=Path("law.txt"), mime_type="text/plain", raw_text=text, metadata={}))
    document.metadata.setdefault("page_spans", [[1, 1]])
    parsed = parse_document(document, build_context())
    subpoint = next(norm for norm in parsed.norms if norm.subpoint == "2")
    start, end = subpoint.span_offsets[0]
    assert document.raw_text[start:end].strip() == subpoint.raw_text.strip()
