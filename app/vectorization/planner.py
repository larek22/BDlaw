"""Automatic generation of vectorization plans."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, cast

from .chunker import apply_plan
from .models import (
    ChunkPreview,
    ChunkingPolicy,
    ChunkingMode,
    DistanceMetric,
    HierarchyRules,
    IdStrategy,
    PayloadSchema,
    PlanPreview,
    QualityChecks,
    VectorizationPlan,
)
from .normalizer import Block
from .tokenization import DEFAULT_TOKENIZER, Tokenizer

try:  # pragma: no cover - optional dependency
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PlanningContext:
    embedding_model: str
    distance: DistanceMetric
    collection_name: str
    file_type: str
    hard_cap: int
    test_queries: Sequence[str] | None = None
    chat_model: str | None = None
    api_key: str | None = None


DEFAULT_PAYLOAD_FIELDS: Dict[str, str] = {
    "source_file": "string",
    "path": "string",
    "lang": "string|null",
    "title_text": "string|null",
    "row_index": "int|null",
}


def build_fallback_plan(
    *,
    blocks: Sequence[Block],
    context: PlanningContext,
) -> VectorizationPlan:
    has_headings = any(block.type == "heading" for block in blocks)
    has_records = any(block.type == "record" for block in blocks)
    if has_records:
        mode: ChunkingMode = "by_records"
    elif has_headings:
        mode = "by_headings"
    else:
        mode = "by_paragraphs"

    split_on_headings = ["h1", "h2", "h3"] if mode == "by_headings" else []
    source_name = Path(blocks[0].path).stem if blocks else "document"

    return VectorizationPlan(
        doc_type=context.file_type,
        collection_name=context.collection_name,
        embedding_model=context.embedding_model,
        distance=context.distance,
        plan_source="fallback",
        id_strategy=IdStrategy(
            pattern=f"auto:{source_name}:{{start}}-{{end}}:{{index}}",
            ensure_uuid_if_missing=True,
        ),
        hierarchy_rules=HierarchyRules(
            heading_regex={
                "h1": r"(?m)^(?:#|\s*h1\b|section)\s+.+$",
                "h2": r"(?m)^(?:##|\s*h2\b|chapter)\s+.+$",
                "h3": r"(?m)^(?:###|\s*h3\b|article)\s+.+$",
            },
            path_fields=["corpus", "part", "chapter", "article", "version"],
        ),
        chunking_policy=ChunkingPolicy(
            mode=cast(ChunkingMode, mode),
            split_on_headings=split_on_headings,
        ),
        payload_schema=PayloadSchema(fields=dict(DEFAULT_PAYLOAD_FIELDS)),
        quality_checks=QualityChecks(),
        test_queries=list(context.test_queries or []),
    )


class GPTPlanner:
    """Wrapper around the OpenAI API to request a plan."""

    def __init__(self, *, model: str, api_key: str) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package not available")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def plan(
        self,
        *,
        blocks: Sequence[Block],
        context: PlanningContext,
    ) -> VectorizationPlan:
        outline = self._build_outline(blocks)
        samples = self._sample_blocks(blocks)
        schema = VectorizationPlan.json_schema()
        system_prompt = (
            "You are an expert pre-ingestion planner for vector databases. "
            "Infer the best vectorization plan from the provided outline and samples. "
            "Return ONLY valid JSON that conforms exactly to the provided schema. No prose."
        )
        user_prompt = self._build_prompt(context, outline, samples, schema)
        response = self.client.responses.create(  # type: ignore[attr-defined]
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        result = response.output[0].content[0].text  # type: ignore[index]
        return self._parse_plan(result)

    def _build_outline(self, blocks: Sequence[Block]) -> str:
        headings = [block.text for block in blocks if block.type == "heading"]
        return "\n".join(headings[:10])

    def _sample_blocks(self, blocks: Sequence[Block]) -> str:
        samples = []
        for block in blocks:
            samples.append(f"[{block.type}] {block.text[:200]}")
            if len(samples) >= 3:
                break
        return "\n".join(samples)

    def _build_prompt(
        self,
        context: PlanningContext,
        outline: str,
        samples: str,
        schema: Dict[str, object],
    ) -> str:
        template = (
            "Context:\n"
            "- Embedding model: {embedding_model}, distance: {distance}, max_tokens hard cap: {hard_cap}\n"
            "- File type: {file_type}\n"
            "- Outline/headings (truncated):\n{outline}\n\n"
            "Representative blocks (truncated):\n{samples}\n\n"
            "Schema (copy exactly):\n{schema}\n\n"
            "Constraints:\n"
            "- Prefer splitting on semantic boundaries.\n"
            "- Keep lists/tables intact when meaningful.\n"
            "- For CSV/JSONL, identify primary key and set record-based chunking.\n"
            "- ID pattern must be stable; add UUID if data missing.\n"
            "- Keep max_tokens <= {hard_cap}; suggest overlap if needed.\n\n"
            "Return the plan JSON only."
        )
        return template.format(
            embedding_model=context.embedding_model,
            distance=context.distance,
            hard_cap=context.hard_cap,
            file_type=context.file_type,
            outline=outline or "(none)",
            samples=samples or "(none)",
            schema=json.dumps(schema, ensure_ascii=False, indent=2),
        )

    def _parse_plan(self, payload: str) -> VectorizationPlan:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON from GPT: {exc}") from exc
        try:
            return VectorizationPlan.from_dict(data)
        except Exception as exc:
            raise RuntimeError(f"Plan validation failed: {exc}") from exc


class VectorizationPlanner:
    """High-level orchestrator for GPT + fallback planning."""

    def __init__(self, *, tokenizer: Tokenizer | None = None) -> None:
        self.tokenizer = tokenizer or DEFAULT_TOKENIZER

    def analyze(
        self,
        *,
        blocks: Sequence[Block],
        context: PlanningContext,
    ) -> PlanPreview:
        plan = self._generate_plan(blocks=blocks, context=context)
        return self._preview(blocks=blocks, plan=plan)

    def _generate_plan(self, *, blocks: Sequence[Block], context: PlanningContext) -> VectorizationPlan:
        planner: GPTPlanner | None = None
        if context.chat_model and context.api_key:
            try:
                planner = GPTPlanner(model=context.chat_model, api_key=context.api_key)
            except Exception as exc:  # pragma: no cover - network initialisation
                logger.warning("Unable to initialise GPT planner: %s", exc)
                planner = None
        if planner is not None:
            try:
                plan = planner.plan(blocks=blocks, context=context)
                return plan.model_copy(update={"plan_source": "gpt"})
            except Exception as exc:
                logger.warning("GPT planning failed, using fallback: %s", exc)
        return build_fallback_plan(blocks=blocks, context=context)

    def _preview(self, *, blocks: Sequence[Block], plan: VectorizationPlan) -> PlanPreview:
        sample_blocks = blocks[: min(len(blocks), 50)]
        chunk_result = apply_plan(sample_blocks, plan, tokenizer=self.tokenizer)
        chunk_previews = [
            ChunkPreview(
                text=chunk.text,
                tokens=chunk.tokens,
                start=chunk.start,
                end=chunk.end,
                meta=chunk.meta,
                id_example=chunk.chunk_id,
            )
            for chunk in chunk_result.chunks[:5]
        ]
        stats = {
            "chunks_estimate": len(chunk_result.chunks),
            "avg_tokens": (
                sum(chunk.tokens for chunk in chunk_result.chunks) / len(chunk_result.chunks)
                if chunk_result.chunks
                else 0
            ),
            "max_tokens": max((chunk.tokens for chunk in chunk_result.chunks), default=0),
            "overlap": plan.chunking_policy.overlap_tokens,
            "embedding_model": plan.embedding_model,
            "distance": plan.distance,
        }
        return PlanPreview(plan=plan, chunks=chunk_previews, stats=stats)


__all__ = [
    "PlanningContext",
    "VectorizationPlanner",
    "GPTPlanner",
    "build_fallback_plan",
]
