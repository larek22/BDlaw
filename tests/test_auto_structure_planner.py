from __future__ import annotations

from app.structure_planner import AutoStructurePlanner


def test_auto_structure_planner_parses_decimal_articles():
    blob = (
        "Статья 783.1 Заголовок\n"
        "Текст первой статьи.\n\n"
        "Статья 860.12 Вторая статья\n"
        "Продолжение текста."
    ).encode("utf-8")

    plan = AutoStructurePlanner.plan(blob, mime="text/plain")

    assert len(plan) == 2
    first = plan[0]
    assert first["hierarchy"]["article_no"] == "783.1"
    assert first["hierarchy"]["article_no_int"] == 7831
    assert "Текст первой" in first["body_text"]

    second = plan[1]
    assert second["hierarchy"]["article_no"] == "860.12"
    assert second["hierarchy"]["article_no_int"] == 86012
