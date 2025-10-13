import pytest

from app.structure_planner import AutoStructurePlanner


def test_auto_structure_schema_contains_required_fields():
    schema = AutoStructurePlanner.JSON_SCHEMA
    assert schema["type"] == "object"
    assert set(schema["required"]) == {"doc_id", "hierarchy", "title_text", "body_text"}
    hierarchy = schema["properties"]["hierarchy"]
    assert {"article_no", "article_no_int"}.issubset(set(hierarchy["required"]))


def test_auto_structure_plan_not_implemented():
    with pytest.raises(NotImplementedError):
        AutoStructurePlanner.plan(b"", "text/plain")
