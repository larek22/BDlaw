from __future__ import annotations

from pathlib import Path

import json

import pytest

from app.vectorization.normalizer import Block
from app.vectorization.planner import (
    PlanningContext,
    get_or_create_plan,
    vectorization_plan_from_dict,
)
from app.vectorization.gpt_planner import GPTStructurePlanner


@pytest.fixture(autouse=True)
def _set_fake_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("GPT_MODEL", "gpt-test")


def _fake_plan() -> dict:
    return {
        "doc_type": "rtf",
        "hierarchy_rules": {
            "heading_regex": {
                "part": "(?i)^часть\\s+\\d+",
                "section": "(?i)^раздел\\s+[ivxlcdm]+",
                "chapter": "(?i)^глава\\s+\\d+",
                "article": "(?i)^статья\\s+\\d+(?:\\.\\d+)?",
            },
            "version_regex": "(?im)от\\s+(\\d{2}\\.\\d{2}\\.\\d{4})",
            "path_fields": ["part", "section", "chapter", "article", "version_date"],
        },
        "preamble_rules": {
            "drop_before_first_heading": None,
            "exclude_regexes": ["(?im)^Оглавление"],
        },
        "chunking_policy": {
            "mode": "by_headings",
            "split_on_headings": ["chapter", "article"],
            "max_tokens": 900,
            "overlap_tokens": 100,
            "keep_lists_intact": True,
            "keep_tables_intact": True,
            "merge_short_paragraphs_under_tokens": 50,
        },
        "quality_checks": {
            "min_chunks": 1,
            "max_tokens_per_chunk": 900,
            "forbid_empty_text": True,
            "dedupe_near_duplicates": True,
        },
    }


def test_get_or_create_plan_uses_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(
        GPTStructurePlanner,
        "generate_plan",
        lambda self, text, filename: _fake_plan(),
    )
    cache_dir = tmp_path / "cache"
    plan_sidecar = tmp_path / "doc_plan.json"
    sample_text = "Статья 1. Пример текста.\nСтатья 2. Продолжение."

    plan = get_or_create_plan(
        sample_text,
        "doc.rtf",
        plan_path=plan_sidecar,
        cache_dir=cache_dir,
    )

    assert plan["chunking_policy"]["max_tokens"] == 900
    assert plan_sidecar.exists()

    cached = get_or_create_plan(
        sample_text,
        "doc.rtf",
        plan_path=plan_sidecar,
        cache_dir=cache_dir,
    )
    assert cached == plan

    with plan_sidecar.open("r", encoding="utf-8") as handle:
        persisted = json.load(handle)
    assert persisted["hierarchy_rules"]["heading_regex"]["article"].startswith("(?i)")


def test_vectorization_plan_from_dict_normalizes(tmp_path, monkeypatch):
    monkeypatch.setattr(
        GPTStructurePlanner,
        "generate_plan",
        lambda self, text, filename: _fake_plan(),
    )
    blocks = [
        Block(
            type="heading",
            text="Статья 1",
            attrs={"level": 3},
            path=str(Path("doc.rtf")),
        ),
        Block(
            type="paragraph",
            text="Основной текст статьи",
            path=str(Path("doc.rtf")),
        ),
    ]
    context = PlanningContext(
        embedding_model="text-embedding-3-large",
        distance="cosine",
        collection_name="test-collection",
        file_type="rtf",
        hard_cap=1200,
    )

    plan_dict = get_or_create_plan(
        "Статья 1. Текст",
        "doc.rtf",
        plan_path=tmp_path / "doc_plan.json",
        cache_dir=tmp_path / "cache",
    )
    plan = vectorization_plan_from_dict(plan_dict, blocks=blocks, context=context)

    assert plan.plan_source == "gpt(normalized)"
    assert plan.chunking_policy.max_tokens == 900
    assert "article" in plan.hierarchy_rules.heading_regex
    assert plan.payload_schema.fields["doc_id"] == "string"
