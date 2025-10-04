from pathlib import Path

from bdlaw.ingest.base import Document
from bdlaw.parser.structure import ParsingContext, parse_document


def build_document(text: str) -> Document:
    return Document(path=Path("law.txt"), mime_type="text/plain", raw_text=text, metadata={"clean_text": text})


def test_parse_simple_article():
    text = """
    Статья 16.1. Заголовок
    Часть 1. Пункт 1. Текст пункта.
    Часть 2. Дополнительный текст.
    """
    doc = build_document(text)
    context = ParsingContext(
        jurisdiction="ru",
        act_type="federal_law",
        law_code="ГК РФ",
        law_full_title="Гражданский кодекс РФ",
        version_id="2024-01-01",
        valid_from="2024-01-01",
    )
    parsed = parse_document(doc, context)
    assert len(parsed.norms) == 2
    assert parsed.norms[0].article == "16.1"
    assert parsed.norms[0].part == "1"
    assert parsed.norms[0].point == "1"
    assert parsed.norms[1].part == "2"


def test_citation_format_includes_version():
    text = "Статья 12. Часть 1. Текст."
    doc = build_document(text)
    context = ParsingContext(
        jurisdiction="ru",
        act_type="federal_law",
        law_code="ГК РФ",
        law_full_title="Гражданский кодекс РФ",
        version_id="2024-02-02",
        valid_from="2024-02-02",
    )
    parsed = parse_document(doc, context)
    assert "2024-02-02" in parsed.norms[0].citation
