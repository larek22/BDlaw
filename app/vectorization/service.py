"""Public service entry point for the vectorization workflow."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .chunker import apply_plan
from .models import DistanceMetric, PlanPreview, VectorizationPlan
from .normalizer import Block, normalize_document
from .planner import PlanningContext, VectorizationPlanner
from .tokenization import DEFAULT_TOKENIZER, Tokenizer


@dataclass(slots=True)
class VectorizationArtifacts:
    plan: VectorizationPlan
    preview: PlanPreview
    blocks: Sequence[Block]


class VectorizationService:
    """Coordinates analysis, preview, and ingestion."""

    def __init__(self, *, tokenizer: Tokenizer | None = None) -> None:
        self.tokenizer = tokenizer or DEFAULT_TOKENIZER
        self.planner = VectorizationPlanner(tokenizer=self.tokenizer)

    def analyze(
        self,
        *,
        path: Path,
        collection_name: str,
        embedding_model: str,
        distance: DistanceMetric,
        hard_cap: int,
        chat_model: str | None = None,
        api_key: str | None = None,
        test_queries: Sequence[str] | None = None,
    ) -> VectorizationArtifacts:
        blocks = normalize_document(path)
        context = PlanningContext(
            embedding_model=embedding_model,
            distance=distance,
            collection_name=collection_name,
            file_type=path.suffix.lower().strip("."),
            hard_cap=hard_cap,
            chat_model=chat_model,
            api_key=api_key,
            test_queries=test_queries,
        )
        preview = self.planner.analyze(blocks=blocks, context=context)
        return VectorizationArtifacts(plan=preview.plan, preview=preview, blocks=blocks)

    def apply_plan(
        self,
        *,
        blocks: Sequence[Block],
        plan: VectorizationPlan,
    ):
        return apply_plan(blocks, plan, tokenizer=self.tokenizer)


__all__ = ["VectorizationService", "VectorizationArtifacts"]
