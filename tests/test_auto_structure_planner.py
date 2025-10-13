from __future__ import annotations

from app.structure_planner import AutoStructurePlanner


def test_auto_structure_planner_parses_articles():
    blob = (
        "Раздел IV\nГлава 32\nСтатья 783.1 Особенности\nТекст статьи\n"
        "Статья 860.12 Дополнительно\nВторой абзац"
    ).encode("utf-8")
    records = AutoStructurePlanner.plan(blob, mime="text/plain", hints={"doc_id": "law-1"})
    assert len(records) == 2
    assert records[0]["hierarchy"]["article_no"] == "783.1"
    assert records[0]["hierarchy"]["article_no_int"] == 7831
    assert records[1]["hierarchy"]["article_no"] == "860.12"
    assert records[1]["hierarchy"]["article_no_int"] == 86012
    for record in records:
        for key in AutoStructurePlanner.JSON_SCHEMA["required"]:
            assert key in record
