"""Automatic generation of vectorization plans."""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple, cast

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
    "doc_id": "string",
    "source_file": "string",
    "path": "string",
    "lang": "string|null",
    "row_index": "int|null",
    "plan_hash": "string|null",
    "plan_source": "string",
}

_RU_ARTICLE_RE = re.compile(r"(?im)^\s*статья\s+\d+")


def _looks_like_russian_code(blocks: Sequence[Block]) -> bool:
    matches = 0
    for block in blocks:
        if block.type != "heading":
            continue
        if _RU_ARTICLE_RE.search(block.text):
            matches += 1
        if matches >= 3:
            return True
    return False


def build_fallback_plan(
    *,
    blocks: Sequence[Block],
    context: PlanningContext,
) -> VectorizationPlan:
    has_headings = any(block.type == "heading" for block in blocks)
    has_records = any(block.type == "record" for block in blocks)
    looks_russian = _looks_like_russian_code(blocks)
    if has_records:
        mode: ChunkingMode = "by_records"
    elif has_headings:
        mode = "by_headings"
    else:
        mode = "by_paragraphs"

    hierarchy_rules = HierarchyRules()
    split_on_headings: list[str] = []
    if mode == "by_headings":
        split_on_headings = ["article", "chapter", "section", "h1", "h2", "h3"]
        if looks_russian:
            hierarchy_rules.heading_regex = {
                "section": r"(?im)^раздел\s+([ivxcl]+|\d+)",
                "part": r"(?im)^часть\s+\d+",
                "chapter": r"(?im)^глава\s+\d+",
                "article": r"(?im)^статья\s+\d+",
            }
            hierarchy_rules.version_regex = r"(?im)от\s+\d{2}\.\d{2}\.\d{4}"
            hierarchy_rules.path_fields = [
                "corpus",
                "section",
                "part",
                "chapter",
                "article",
                "version",
            ]
            split_on_headings = ["article"]

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
        hierarchy_rules=hierarchy_rules,
        chunking_policy=ChunkingPolicy(
            mode=cast(ChunkingMode, mode),
            max_tokens=min(context.hard_cap, 1200),
            overlap_tokens=120,
            split_on_headings=split_on_headings,
        ),
        payload_schema=PayloadSchema(fields=dict(DEFAULT_PAYLOAD_FIELDS)),
        quality_checks=QualityChecks(max_tokens_per_chunk=context.hard_cap),
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
        result = self._call_model(system_prompt, user_prompt)
        try:
            return self._parse_plan(result)
        except RuntimeError as exc:
            logger.debug("GPT plan invalid, attempting repair: %s", exc)
            repair_prompt = self._build_repair_prompt(schema, str(exc))
            repaired = self._call_model(system_prompt, repair_prompt)
            return self._parse_plan(repaired)

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

    def _build_repair_prompt(self, schema: Dict[str, object], error_message: str) -> str:
        return (
            f"Your previous JSON was invalid: {error_message}.\n"
            f"Schema (copy exactly):\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n"
            "Return ONLY valid JSON that matches the schema exactly. No prose. No comments."
        )

    def _call_model(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.responses.create(  # type: ignore[attr-defined]
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        try:
            return response.output[0].content[0].text  # type: ignore[index]
        except Exception as exc:  # pragma: no cover - defensive guard
            raise RuntimeError("Unexpected GPT response structure") from exc

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
        self._plan_cache: Dict[Tuple[str, str, str], Dict[str, object]] = {}

    def analyze(
        self,
        *,
        blocks: Sequence[Block],
        context: PlanningContext,
    ) -> PlanPreview:
        plan = self._generate_plan(blocks=blocks, context=context)
        return self.preview_from_plan(blocks=blocks, plan=plan)

    def preview_from_plan(
        self,
        *,
        blocks: Sequence[Block],
        plan: VectorizationPlan,
        sample_size: int = 5,
    ) -> PlanPreview:
        return self._preview(blocks=blocks, plan=plan, sample_size=sample_size)

    def _generate_plan(self, *, blocks: Sequence[Block], context: PlanningContext) -> VectorizationPlan:
        cache_key = self._cache_key(blocks, context)
        cached = self._plan_cache.get(cache_key)
        if cached is not None:
            return VectorizationPlan.from_dict(json.loads(json.dumps(cached)))
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
                plan = plan.model_copy(update={"plan_source": "gpt"})
                self._plan_cache[cache_key] = plan.model_dump()
                return plan
            except Exception as exc:
                logger.warning("GPT planning failed, using fallback: %s", exc)
        plan = build_fallback_plan(blocks=blocks, context=context)
        self._plan_cache[cache_key] = plan.model_dump()
        return plan

    def cache_info(self) -> Dict[str, int]:
        return {"entries": len(self._plan_cache)}

    def _cache_key(self, blocks: Sequence[Block], context: PlanningContext) -> Tuple[str, str, str]:
        return (
            self._hash_blocks(blocks),
            context.file_type,
            context.embedding_model,
        )

    def _hash_blocks(self, blocks: Sequence[Block]) -> str:
        hasher = hashlib.sha256()
        for block in blocks:
            hasher.update(block.type.encode("utf-8", "ignore"))
            hasher.update(b"\0")
            hasher.update(block.text.encode("utf-8", "ignore"))
            hasher.update(b"\0")
        return hasher.hexdigest()

    def _preview(
        self,
        *,
        blocks: Sequence[Block],
        plan: VectorizationPlan,
        sample_size: int = 5,
    ) -> PlanPreview:
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
            for chunk in chunk_result.chunks[:sample_size]
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
