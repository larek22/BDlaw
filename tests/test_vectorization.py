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
from app.vectorization.models import DistanceMetric
from app.vectorization.normalizer import Block
from app.vectorization.planner import PlanningContext, VectorizationPlanner, build_fallback_plan
from app.vectorization.quality import run_quality_checks
from app.vectorization.service import VectorizationService


@pytest.fixture
def sample_blocks(tmp_path: Path) -> list[Block]:
    path = tmp_path / "sample.txt"
    path.write_text("# Heading\nParagraph one.\nParagraph two.", encoding="utf-8")
    return [
        Block(type="heading", text="Heading", attrs={"level": 1}, path=str(path), position={"start": 0, "end": 7}),
        Block(type="paragraph", text="Paragraph one.", attrs={}, path=str(path), position={"start": 8, "end": 22}),
        Block(type="paragraph", text="Paragraph two.", attrs={}, path=str(path), position={"start": 23, "end": 37}),
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
