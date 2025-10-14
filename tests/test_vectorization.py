from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from app.vectorization.chunker import apply_plan
from app.vectorization.models import DistanceMetric
from app.vectorization.normalizer import Block
from app.vectorization.planner import PlanningContext, build_fallback_plan
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
