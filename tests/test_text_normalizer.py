from pathlib import Path

from bdlaw.ingest.base import Document
from bdlaw.normalize import TextNormalizer, normalize_document


def _document(text: str) -> Document:
    return Document(path=Path("law.txt"), mime_type="text/plain", raw_text=text, metadata={})


def test_basic_normalization_replaces_nbsp_and_quotes():
    text = "Статья 1\xa0«Тест» ‘пример’.\nСтр. 1"
    document = normalize_document(_document(text))
    clean = document.metadata["clean_text"]
    assert "\xa0" not in clean
    assert "«" not in clean and '"' in clean
    stats = document.metadata["normalization_stats"]
    assert stats["normalize_quotes"] >= 1
    assert any(key.startswith("header_footer") for key in stats)


def test_hyphenated_words_and_staircase_removed():
    text = "Сло-\nво должно быть\n    без лесенок."
    document = normalize_document(_document(text))
    clean = document.metadata["clean_text"]
    assert "Слово" in clean
    assert "\n    " not in clean


def test_optional_yo_replacement():
    normalizer = TextNormalizer(replace_yo=True)
    document = normalizer(_document("ёжик в тумане"))
    clean = document.metadata["clean_text"]
    assert "ё" not in clean
    assert document.metadata["yo_normalized"] is True
