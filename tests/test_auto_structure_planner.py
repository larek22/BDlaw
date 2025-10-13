from app.structure_planner import AutoStructurePlanner


def test_auto_structure_planner_extracts_articles_with_decimals():
    blob = (
        "Раздел I\n"
        "Глава 2\n"
        "Статья 783.1 Заголовок первой статьи\n"
        "Текст первой статьи.\n"
        "Статья 860.12 Заголовок второй статьи\n"
        "Текст второй статьи."
    ).encode("utf-8")

    records = AutoStructurePlanner.plan(blob, "text/plain", hints={"doc_id": "gk_rf"})

    assert len(records) == 2
    numbers = {item["hierarchy"]["article_no"] for item in records}
    assert {"783.1", "860.12"} == numbers
    bases = {item["hierarchy"]["article_no_int"] for item in records}
    assert bases == {783, 860}
    for item in records:
        assert item["title_text"]
        assert item["body_text"]
