from bdlaw.parser.amendments import extract_amendments
from bdlaw.parser.annotations import extract_annotations
from bdlaw.parser.crossrefs import extract_cross_refs


def test_extract_amendments_returns_clean_text_and_payload():
    text = "Статья 1 (в ред. Федерального закона от 30.12.2012 № 302-ФЗ)."
    cleaned, amendments = extract_amendments(text)
    assert "в ред." not in cleaned
    assert amendments == [{"date": "2012-12-30", "law_no": "302-ФЗ", "scope": "article"}]


def test_extract_annotations():
    text = "(Постановление Конституционного суда РФ от 03.07.2019 № 26-П https://example.com)."
    cleaned, annotations = extract_annotations(text)
    assert not cleaned
    assert annotations[0]["type"] == "cc_ruling"
    assert annotations[0]["source_url"] == "https://example.com"


def test_extract_cross_refs():
    text = "см. ст. 12 ГК РФ, по ст. 395 ч. 1"
    refs = extract_cross_refs(text, default_law="ГК РФ")
    assert any(ref["article"] == "12" for ref in refs)
    assert any(ref["part"] == "1" for ref in refs)
