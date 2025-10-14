from __future__ import annotations

from pathlib import Path
from typing import Sequence, cast

import pytest

from app.vectorization.auto_pipeline import (
    AutoPipelineConfig,
    AutoPipelineDeps,
    AutoVectorizationPipeline,
)
from app.vectorization.chunker import ChunkerResult, apply_plan
from app.vectorization.models import DistanceMetric, VectorizationPlan
from app.vectorization.normalizer import Block
from app.vectorization.planner import (
    PlanningContext,
    VectorizationPlanner,
    build_fallback_plan,
    _normalize_plan_dict,
)
from app.vectorization.quality import run_quality_checks
from app.vectorization.service import VectorizationService, filter_blocks_for_plan


@pytest.fixture
def sample_blocks(tmp_path: Path) -> list[Block]:
    path = tmp_path / "sample.txt"
    path.write_text("# Heading\nParagraph one.\nParagraph two.", encoding="utf-8")
    return [
        Block(type="heading", text="Heading", attrs={"level": 1}, path=str(path), position={"start": 0, "end": 7}),
        Block(type="paragraph", text="Paragraph one.", attrs={}, path=str(path), position={"start": 8, "end": 22}),
        Block(type="paragraph", text="Paragraph two.", attrs={}, path=str(path), position={"start": 23, "end": 37}),
    ]


@pytest.fixture
def russian_blocks(tmp_path: Path) -> list[Block]:
    path = tmp_path / "ru.txt"
    path.write_text("stub", encoding="utf-8")
    return [
        Block(
            type="paragraph",
            text="Принят Государственной Думой",
            attrs={},
            path=str(path),
            position={"start": 0, "end": 30},
        ),
        Block(
            type="heading",
            text="Статья 1. Общие положения",
            attrs={"level": 3},
            path=str(path),
            position={"start": 31, "end": 60},
        ),
        Block(
            type="paragraph",
            text="Текст статьи 1",
            attrs={},
            path=str(path),
            position={"start": 61, "end": 80},
        ),
        Block(
            type="heading",
            text="Статья 2. Полномочия",
            attrs={"level": 3},
            path=str(path),
            position={"start": 81, "end": 110},
        ),
        Block(
            type="paragraph",
            text="Текст статьи 2",
            attrs={},
            path=str(path),
            position={"start": 111, "end": 130},
        ),
        Block(
            type="heading",
            text="Статья 3. Заключительные положения",
            attrs={"level": 3},
            path=str(path),
            position={"start": 131, "end": 180},
        ),
    ]


def test_build_fallback_plan(sample_blocks: list[Block]) -> None:
    context = PlanningContext(
        embedding_model="text-embedding-3-small",
        distance="cosine",
        collection_name="demo",
        file_type="txt",
        hard_cap=1200,
    )
    plan = build_fallback_plan(blocks=sample_blocks, context=context)
    assert plan.plan_source == "fallback"
    assert plan.chunking_policy.mode == "by_headings"
    assert plan.id_strategy.ensure_uuid_if_missing is True
    assert "article" in plan.chunking_policy.split_on_headings


def test_ru_fallback_article_split(russian_blocks: list[Block]) -> None:
    context = PlanningContext(
        embedding_model="text-embedding-3-small",
        distance="cosine",
        collection_name="demo",
        file_type="rtf",
        hard_cap=1000,
    )
    plan = build_fallback_plan(blocks=russian_blocks, context=context)
    assert plan.chunking_policy.mode == "by_headings"
    assert plan.chunking_policy.split_on_headings == ["article"]
    assert plan.chunking_policy.max_tokens == 900
    assert plan.preamble_rules.drop_before_first_heading == "article"
    assert any("Статья" in block.text for block in russian_blocks)


def test_apply_plan_deterministic(sample_blocks: list[Block]) -> None:
    context = PlanningContext(
        embedding_model="text-embedding-3-small",
        distance="cosine",
        collection_name="demo",
        file_type="txt",
        hard_cap=1200,
    )
    plan = build_fallback_plan(blocks=sample_blocks, context=context)
    result_first = apply_plan(sample_blocks, plan)
    result_second = apply_plan(sample_blocks, plan)
    assert [chunk.chunk_id for chunk in result_first.chunks] == [
        chunk.chunk_id for chunk in result_second.chunks
    ]
    assert [chunk.text for chunk in result_first.chunks] == [chunk.text for chunk in result_second.chunks]


def test_apply_plan_uuid5_ids(sample_blocks: list[Block]) -> None:
    plan = build_fallback_plan(
        blocks=sample_blocks,
        context=PlanningContext(
            embedding_model="text-embedding-3-small",
            distance="cosine",
            collection_name="demo",
            file_type="txt",
            hard_cap=1200,
        ),
    )
    plan.id_strategy.pattern = "auto"
    result_first = apply_plan(sample_blocks, plan)
    result_second = apply_plan(sample_blocks, plan)
    assert [chunk.chunk_id for chunk in result_first.chunks] == [
        chunk.chunk_id for chunk in result_second.chunks
    ]


def test_token_cap_and_ranges(sample_blocks: list[Block]) -> None:
    plan = build_fallback_plan(
        blocks=sample_blocks,
        context=PlanningContext(
            embedding_model="text-embedding-3-small",
            distance="cosine",
            collection_name="demo",
            file_type="txt",
            hard_cap=50,
        ),
    )
    plan.chunking_policy.max_tokens = 10
    plan.chunking_policy.overlap_tokens = 0
    result = apply_plan(sample_blocks, plan)
    assert result.chunks
    tokens = [chunk.tokens for chunk in result.chunks]
    assert max(tokens) <= plan.chunking_policy.max_tokens
    starts = [chunk.start for chunk in result.chunks]
    ends = [chunk.end for chunk in result.chunks]
    assert starts == sorted(starts)
    assert ends == sorted(ends)
    assert len({(chunk.start, chunk.end) for chunk in result.chunks}) == len(result.chunks)


def test_filter_blocks_skips_preamble(russian_blocks: list[Block]) -> None:
    context = PlanningContext(
        embedding_model="text-embedding-3-small",
        distance="cosine",
        collection_name="demo",
        file_type="rtf",
        hard_cap=900,
    )
    plan = build_fallback_plan(blocks=russian_blocks, context=context)
    filtered = filter_blocks_for_plan(russian_blocks, plan)
    assert filtered[0].text.startswith("Статья 1")
    assert all("Принят" not in block.text for block in filtered)


def test_vectorization_service_analysis(tmp_path: Path) -> None:
    path = tmp_path / "doc.txt"
    path.write_text("Heading\nParagraph", encoding="utf-8")
    service = VectorizationService()
    artifacts = service.analyze(
        path=path,
        collection_name="demo",
        embedding_model="text-embedding-3-small",
        distance=cast(DistanceMetric, "cosine"),
        hard_cap=600,
    )
    assert artifacts.plan.collection_name == "demo"
    assert artifacts.preview.chunks


def test_normalize_plan_remaps_payload_fields(sample_blocks: list[Block]) -> None:
    plan_data = {
        "payload_schema": {"fields": {"title": "string", "custom": "int"}},
        "chunking_policy": {"max_tokens": 500},
    }
    context = PlanningContext(
        embedding_model="text-embedding-3-small",
        distance="cosine",
        collection_name="demo",
        file_type="txt",
        hard_cap=500,
    )
    normalized = _normalize_plan_dict(plan_data, blocks=sample_blocks, context=context)
    fields = normalized["payload_schema"]["fields"]
    assert "title" not in fields
    assert fields["title_text"] == "string"
    assert "custom" not in fields
    assert fields["hierarchy"] == "object"
    assert fields["law_meta"] == "object"


def test_ru_chunks_include_hierarchy_and_titles(russian_blocks: list[Block]) -> None:
    context = PlanningContext(
        embedding_model="text-embedding-3-small",
        distance="cosine",
        collection_name="demo",
        file_type="rtf",
        hard_cap=900,
    )
    plan = build_fallback_plan(blocks=russian_blocks, context=context)
    filtered = filter_blocks_for_plan(russian_blocks, plan)
    result = apply_plan(filtered, plan)
    assert result.chunks
    first = result.chunks[0]
    assert first.meta.get("title_text", "").startswith("Статья 1")
    hierarchy = first.meta.get("hierarchy") or {}
    assert hierarchy.get("article_no_int") == 1
    assert first.meta.get("lang") == "ru"
    assert all(chunk.tokens <= plan.chunking_policy.max_tokens for chunk in result.chunks)


def test_planner_cache_reuse(sample_blocks: list[Block]) -> None:
    planner = VectorizationPlanner()
    context = PlanningContext(
        embedding_model="text-embedding-3-small",
        distance="cosine",
        collection_name="demo",
        file_type="txt",
        hard_cap=1200,
    )
    first = planner.analyze(blocks=sample_blocks, context=context)
    assert planner.cache_info()["entries"] == 1
    second = planner.analyze(blocks=sample_blocks, context=context)
    assert planner.cache_info()["entries"] == 1
    assert first.plan.model_dump() == second.plan.model_dump()


def test_planner_allowlist_retry(monkeypatch, sample_blocks: list[Block]) -> None:
    calls: list[str] = []

    class DummyPlanner:
        def __init__(self, *, models, api_key: str) -> None:
            self.models = list(models)
            self.last_model_used = None

        def plan(self, *, blocks, context) -> VectorizationPlan:  # type: ignore[override]
            for model in self.models:
                calls.append(model)
                if model == "gpt-4.1":
                    continue
                self.last_model_used = model
                return build_fallback_plan(blocks=blocks, context=context).model_copy(
                    update={"plan_source": "gpt(normalized)", "planner_model": model}
                )
            raise RuntimeError("model not allowed")

    monkeypatch.setattr(
        "app.vectorization.planner.GPTPlanner",
        DummyPlanner,
    )

    planner = VectorizationPlanner()
    context = PlanningContext(
        embedding_model="text-embedding-3-small",
        distance="cosine",
        collection_name="demo",
        file_type="txt",
        hard_cap=600,
        chat_model="gpt-4.1",
        api_key="test",
    )
    preview = planner.analyze(blocks=sample_blocks, context=context)
    assert preview.plan.plan_source == "gpt(normalized)"
    assert calls == ["gpt-4.1", "gpt-4o-mini"]


def test_planner_ru_fallback_when_models_fail(monkeypatch, russian_blocks: list[Block]) -> None:
    class FailingPlanner:
        def __init__(self, *, models, api_key: str) -> None:
            self.models = list(models)
            self.last_model_used = None

        def plan(self, *, blocks, context):  # type: ignore[override]
            raise RuntimeError("boom")

    monkeypatch.setattr(
        "app.vectorization.planner.GPTPlanner",
        FailingPlanner,
    )

    planner = VectorizationPlanner()
    context = PlanningContext(
        embedding_model="text-embedding-3-small",
        distance="cosine",
        collection_name="demo",
        file_type="rtf",
        hard_cap=900,
        chat_model="gpt-4.1",
        api_key="key",
    )
    preview = planner.analyze(blocks=russian_blocks, context=context)
    plan = preview.plan
    assert plan.plan_source == "fallback-ru"
    assert plan.chunking_policy.split_on_headings == ["article"]


def test_quality_checks_low_coverage(sample_blocks: list[Block]) -> None:
    context = PlanningContext(
        embedding_model="text-embedding-3-small",
        distance="cosine",
        collection_name="demo",
        file_type="txt",
        hard_cap=1200,
    )
    plan = build_fallback_plan(blocks=sample_blocks, context=context)
    empty_result = ChunkerResult(chunks=[], token_usage=0)
    report = run_quality_checks(sample_blocks, empty_result, plan, coverage_threshold=0.95)
    assert not report.ok
    assert any("coverage" in issue.lower() for issue in report.issues)


def test_auto_pipeline_auto_success(tmp_path: Path) -> None:
    doc = tmp_path / "auto.txt"
    doc.write_text("# Title\nBody paragraph", encoding="utf-8")
    output_dir = tmp_path / "manifests"
    upserts: list[tuple[list[str], str]] = []
    alias_calls: list[str] = []

    def embed(texts: Sequence[str], model: str) -> Sequence[Sequence[float]]:
        return [[float(idx)] for idx, _ in enumerate(texts)]

    def upsert(chunks, vectors, plan, plan_hash) -> None:
        upserts.append(([chunk.chunk_id for chunk in chunks], plan_hash))

    def alias_swap(collection: str) -> None:
        alias_calls.append(collection)

    pipeline = AutoVectorizationPipeline()
    config = AutoPipelineConfig(collection_name="demo", output_dir=output_dir)
    deps = AutoPipelineDeps(embed_batch=embed, upsert_batch=upsert, alias_swap=alias_swap)
    result = pipeline.process_paths([doc], config=config, deps=deps)[0]

    assert result.ingested is True
    assert result.escalated is False
    assert upserts
    assert alias_calls == ["demo"]
    plan_path = output_dir / "demo" / f"{doc.stem}.plan.json"
    assert plan_path.exists()


def test_auto_pipeline_escalates_without_dependencies(tmp_path: Path) -> None:
    doc = tmp_path / "auto_escalate.txt"
    doc.write_text("# Title\nParagraph", encoding="utf-8")
    config = AutoPipelineConfig(collection_name="demo", output_dir=tmp_path / "manifests")
    pipeline = AutoVectorizationPipeline()
    result = pipeline.process_paths([doc], config=config)[0]
    assert result.ingested is False
    assert result.escalated is True
