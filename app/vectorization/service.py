"""Public service entry point for the vectorization workflow."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import re

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
        filtered_blocks = filter_blocks_for_plan(blocks, preview.plan)
        refined_preview = self.planner.preview_from_plan(
            blocks=filtered_blocks, plan=preview.plan
        )
        return VectorizationArtifacts(
            plan=refined_preview.plan, preview=refined_preview, blocks=filtered_blocks
        )

    def apply_plan(
        self,
        *,
        blocks: Sequence[Block],
        plan: VectorizationPlan,
    ):
        filtered = filter_blocks_for_plan(blocks, plan)
        return apply_plan(filtered, plan, tokenizer=self.tokenizer)


def filter_blocks_for_plan(blocks: Sequence[Block], plan: VectorizationPlan) -> list[Block]:
    """Remove front-matter according to the plan's preamble rules."""

    if not blocks:
        return []

    rules = plan.preamble_rules
    filtered = list(blocks)

    drop_heading = rules.drop_before_first_heading
    if drop_heading:
        target_regex = None
        heading_level: int | None = None
        hierarchy_regex = plan.hierarchy_rules.heading_regex or {}
        if drop_heading in hierarchy_regex and hierarchy_regex[drop_heading]:
            try:
                target_regex = re.compile(hierarchy_regex[drop_heading])
            except re.error:
                target_regex = None
        elif drop_heading.startswith("h") and drop_heading[1:].isdigit():
            heading_level = int(drop_heading[1:])
        else:
            try:
                target_regex = re.compile(drop_heading, re.IGNORECASE)
            except re.error:
                target_regex = None

        start_index = 0
        for idx, block in enumerate(filtered):
            if block.type != "heading":
                continue
            text = block.text or ""
            matched = False
            if target_regex and target_regex.search(text):
                matched = True
            elif heading_level is not None:
                level_value = block.attrs.get("level") if isinstance(block.attrs, dict) else None
                try:
                    level_int = int(level_value) if level_value is not None else None
                except (TypeError, ValueError):
                    level_int = None
                if level_int == heading_level:
                    matched = True
            if matched:
                start_index = idx
                break
        if start_index:
            filtered = filtered[start_index:]

    patterns = []
    for pattern_text in rules.exclude_regexes:
        if not pattern_text:
            continue
        try:
            patterns.append(re.compile(pattern_text))
        except re.error:
            continue

    leading_drop = 0
    consumed_chars = 0
    allow_frontmatter_trim = bool(drop_heading) or bool(patterns)
    for block in filtered:
        if block.type == "heading":
            break
        text = block.text or ""
        should_drop = False
        if patterns and any(pattern.search(text) for pattern in patterns):
            should_drop = True
        elif allow_frontmatter_trim and rules.max_frontmatter_chars > 0:
            consumed_chars += len(text)
            if consumed_chars <= rules.max_frontmatter_chars:
                should_drop = True
        if should_drop:
            leading_drop += 1
            continue
        break
    if leading_drop:
        filtered = filtered[leading_drop:]

    return filtered


__all__ = ["VectorizationService", "VectorizationArtifacts", "filter_blocks_for_plan"]
