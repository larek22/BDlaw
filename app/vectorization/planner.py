"""Automatic generation of vectorization plans."""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple, cast

from ..llm_utils import LLM_MODEL_CANDIDATES, call_llm_with_retries
from .chunker import apply_plan
from .models import (
    ChunkPreview,
    ChunkingPolicy,
    ChunkingMode,
    DistanceMetric,
    HierarchyRules,
    IdStrategy,
    PlanSource,
    PreambleRules,
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

ALLOWED_PLANNER_MODELS: Tuple[str, ...] = LLM_MODEL_CANDIDATES


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
    "lang": "string|null",
    "title_text": "string",
    "body_text": "string",
    "hierarchy": "object",
    "law_meta": "object",
    "row_index": "int|null",
}

_RU_ARTICLE_RE = re.compile(r"(?im)^\s*статья\s+\d+")


def _looks_like_russian_code(blocks: Sequence[Block]) -> bool:
    matches = 0
    for block in blocks:
        if _RU_ARTICLE_RE.search(block.text):
            matches += 1
        if matches >= 3:
            return True
    return False


def _normalize_plan_dict(
    plan_data: Dict[str, object],
    *,
    blocks: Sequence[Block],
    context: PlanningContext,
) -> Dict[str, object]:
    """Sanitise GPT plans to align with our schema and expectations."""

    cloned = json.loads(json.dumps(plan_data)) if plan_data else {}
    looks_russian = _looks_like_russian_code(blocks)

    cloned["doc_type"] = context.file_type
    cloned["collection_name"] = context.collection_name
    cloned["embedding_model"] = context.embedding_model
    cloned["distance"] = context.distance

    payload_schema = cloned.get("payload_schema")
    if not isinstance(payload_schema, dict):
        payload_schema = {}
    fields = payload_schema.get("fields")
    if not isinstance(fields, dict):
        fields = {}
    else:
        fields = dict(fields)
    if "title" in fields and "title_text" not in fields:
        fields["title_text"] = fields.pop("title")
    if "body" in fields and "body_text" not in fields:
        fields["body_text"] = fields.pop("body")
    sanitised_fields: Dict[str, str] = {}
    for key, value in fields.items():
        if key in DEFAULT_PAYLOAD_FIELDS:
            sanitised_fields[key] = str(value)
    for key, dtype in DEFAULT_PAYLOAD_FIELDS.items():
        sanitised_fields.setdefault(key, dtype)
    payload_schema_clean: Dict[str, object] = {"fields": sanitised_fields}
    cloned["payload_schema"] = payload_schema_clean

    id_strategy = cloned.get("id_strategy")
    if not isinstance(id_strategy, dict):
        id_strategy = {}
    pattern = id_strategy.get("pattern")
    if not isinstance(pattern, str) or not pattern.strip():
        pattern = "auto:{start}-{end}:{index}"
    ensure_uuid = bool(id_strategy.get("ensure_uuid_if_missing", True))
    id_strategy_clean = {
        "pattern": pattern,
        "ensure_uuid_if_missing": ensure_uuid,
    }
    cloned["id_strategy"] = id_strategy_clean

    policy = cloned.get("chunking_policy")
    if not isinstance(policy, dict):
        policy = {}
    mode = policy.get("mode")
    has_headings = any(block.type == "heading" for block in blocks)
    if looks_russian:
        policy["mode"] = "by_headings"
    elif mode not in {"by_headings", "by_paragraphs", "by_records"}:
        policy["mode"] = "by_headings" if has_headings else "by_paragraphs"
    max_tokens = policy.get("max_tokens")
    try:
        max_tokens_int = int(max_tokens)
    except (TypeError, ValueError):
        max_tokens_int = context.hard_cap
    if looks_russian:
        max_tokens_int = min(max_tokens_int, 900)
    max_tokens_int = max(1, min(max_tokens_int, context.hard_cap))
    overlap = policy.get("overlap_tokens", 120)
    try:
        overlap_int = int(overlap)
    except (TypeError, ValueError):
        overlap_int = 120
    if overlap_int >= max_tokens_int:
        overlap_int = max(0, min(120, max_tokens_int - 1))
    overlap_int = max(0, overlap_int)
    split_on = policy.get("split_on_headings")
    if isinstance(split_on, (tuple, set)):
        split_on = list(split_on)
    elif isinstance(split_on, list):
        split_on = [str(item) for item in split_on if item]
    elif split_on:
        split_on = [str(split_on)]
    else:
        split_on = []
    if looks_russian:
        split_on = ["article"]
    merge_short = policy.get("merge_short_paragraphs_under_tokens", 80)
    try:
        merge_short_int = int(merge_short)
    except (TypeError, ValueError):
        merge_short_int = 80
    keep_lists = bool(policy.get("keep_lists_intact", True))
    keep_tables = bool(policy.get("keep_tables_intact", True))

    record_policy_raw = policy.get("record_chunking")
    record_policy: Dict[str, object] = {}
    if isinstance(record_policy_raw, dict):
        primary_key = record_policy_raw.get("primary_key")
        if primary_key is not None:
            record_policy["primary_key"] = str(primary_key) if primary_key else None
        include = record_policy_raw.get("fields_include")
        if isinstance(include, list):
            record_policy["fields_include"] = [str(item) for item in include if str(item)]
        exclude = record_policy_raw.get("fields_exclude")
        if isinstance(exclude, list):
            record_policy["fields_exclude"] = [str(item) for item in exclude if str(item)]
        records_per_chunk = record_policy_raw.get("records_per_chunk")
        try:
            record_policy["records_per_chunk"] = max(1, int(records_per_chunk))
        except (TypeError, ValueError):
            pass

    policy_clean: Dict[str, object] = {
        "mode": policy["mode"],
        "max_tokens": max_tokens_int,
        "overlap_tokens": overlap_int,
        "split_on_headings": split_on,
        "keep_lists_intact": keep_lists,
        "keep_tables_intact": keep_tables,
        "merge_short_paragraphs_under_tokens": merge_short_int,
    }
    if record_policy:
        policy_clean["record_chunking"] = record_policy
    cloned["chunking_policy"] = policy_clean

    hierarchy = cloned.get("hierarchy_rules")
    if not isinstance(hierarchy, dict):
        hierarchy = {}
    heading_regex = hierarchy.get("heading_regex")
    if not isinstance(heading_regex, dict):
        heading_regex = {}
    else:
        heading_regex = {
            str(key): str(value)
            for key, value in heading_regex.items()
            if isinstance(key, str) and isinstance(value, str)
        }
    if looks_russian:
        heading_regex.setdefault(
            "section", r"(?im)^\s*Раздел\s+([IVXLC]+|\d+)\b"
        )
        heading_regex.setdefault("part", r"(?im)^\s*Часть\s+(\d+)\b")
        heading_regex.setdefault("chapter", r"(?im)^\s*Глава\s+(\d+)\b")
        heading_regex.setdefault(
            "article", r"(?im)^\s*(Статья|Ст\.)\s+(\d+)\b"
        )
        hierarchy.setdefault(
            "version_regex", r"(?im)от\s+(\d{2}\.\d{2}\.\d{4})"
        )
        hierarchy.setdefault(
            "path_fields",
            ["section", "part", "chapter", "article", "version_date"],
        )
    else:
        hierarchy.setdefault(
            "path_fields",
            hierarchy.get("path_fields")
            or ["corpus", "section", "part", "chapter", "article", "version"],
        )
    path_fields = hierarchy.get("path_fields")
    if isinstance(path_fields, list):
        path_fields_list = [str(item) for item in path_fields if str(item)]
    else:
        path_fields_list = hierarchy.get("path_fields") or [
            "corpus",
            "section",
            "part",
            "chapter",
            "article",
            "version",
        ]
    if looks_russian:
        path_fields_list = ["section", "part", "chapter", "article", "version_date"]
    hierarchy_clean: Dict[str, object] = {
        "heading_regex": heading_regex,
        "path_fields": path_fields_list,
    }
    version_regex = hierarchy.get("version_regex")
    if version_regex is not None:
        if isinstance(version_regex, str):
            hierarchy_clean["version_regex"] = version_regex
    elif looks_russian:
        hierarchy_clean["version_regex"] = r"(?im)от\s+(\d{2}\.\d{2}\.\d{4})"
    cloned["hierarchy_rules"] = hierarchy_clean

    preamble = cloned.get("preamble_rules")
    if not isinstance(preamble, dict):
        preamble = {}
    drop_heading = preamble.get("drop_before_first_heading")
    if looks_russian:
        preamble["drop_before_first_heading"] = "article"
    elif drop_heading is not None and not isinstance(drop_heading, str):
        preamble["drop_before_first_heading"] = None
    exclude = preamble.get("exclude_regexes")
    if isinstance(exclude, list):
        exclude_list = [pattern for pattern in exclude if isinstance(pattern, str)]
    else:
        exclude_list = []
    if looks_russian:
        for pattern in [
            r"(?im)^\(в ред\.",
            r"(?im)^Принят\s+Государственной Думой",
            r"(?im)^Одобрен\s+Советом Федерации",
            r"(?im)^Оглавление\b",
        ]:
            if pattern not in exclude_list:
                exclude_list.append(pattern)
    preamble_clean: Dict[str, object] = {
        "drop_before_first_heading": preamble.get("drop_before_first_heading"),
        "exclude_regexes": exclude_list,
    }
    max_frontmatter = preamble.get("max_frontmatter_chars", 8000)
    try:
        max_frontmatter_int = int(max_frontmatter)
    except (TypeError, ValueError):
        max_frontmatter_int = 8000
    preamble_clean["max_frontmatter_chars"] = max(0, max_frontmatter_int)
    if looks_russian:
        preamble_clean["drop_before_first_heading"] = "article"
    cloned["preamble_rules"] = preamble_clean

    quality = cloned.get("quality_checks")
    if not isinstance(quality, dict):
        quality = {}
    min_chunks = quality.get("min_chunks", 1)
    try:
        min_chunks_int = max(1, int(min_chunks))
    except (TypeError, ValueError):
        min_chunks_int = 1
    max_tokens_per_chunk = quality.get("max_tokens_per_chunk")
    try:
        max_tokens_per_chunk_int = int(max_tokens_per_chunk)
    except (TypeError, ValueError):
        max_tokens_per_chunk_int = policy["max_tokens"]
    max_tokens_per_chunk_int = min(max_tokens_per_chunk_int, policy["max_tokens"])
    quality_clean: Dict[str, object] = {
        "min_chunks": min_chunks_int,
        "max_tokens_per_chunk": max_tokens_per_chunk_int,
        "forbid_empty_text": bool(quality.get("forbid_empty_text", True)),
        "dedupe_near_duplicates": bool(
            quality.get("dedupe_near_duplicates", True)
        ),
    }
    cloned["quality_checks"] = quality_clean

    test_queries = cloned.get("test_queries")
    if not isinstance(test_queries, list):
        test_queries = []
    else:
        test_queries = [str(item) for item in test_queries if isinstance(item, str)]
    cloned["test_queries"] = test_queries

    cloned["plan_source"] = "gpt(normalized)"
    return cloned


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
    preamble_rules = PreambleRules()
    split_on_headings: list[str] = []
    max_tokens = min(context.hard_cap, 1200)
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
            preamble_rules = PreambleRules(
                drop_before_first_heading="article",
                exclude_regexes=[
                    r"(?im)^\(в ред\.",
                    r"(?im)^принят\s+государственной думой",
                    r"(?im)^одобрен\s+советом федерации",
                    r"(?im)^оглавление\b",
                ],
                max_frontmatter_chars=8000,
            )
            max_tokens = min(context.hard_cap, 900)

    source_name = Path(blocks[0].path).stem if blocks else "document"

    plan_source: PlanSource = "fallback"
    if looks_russian and mode == "by_headings":
        plan_source = "fallback-ru"

    return VectorizationPlan(
        doc_type=context.file_type,
        collection_name=context.collection_name,
        embedding_model=context.embedding_model,
        distance=context.distance,
        plan_source=plan_source,
        id_strategy=IdStrategy(
            pattern=f"auto:{source_name}:{{start}}-{{end}}:{{index}}",
            ensure_uuid_if_missing=True,
        ),
        hierarchy_rules=hierarchy_rules,
        preamble_rules=preamble_rules,
        chunking_policy=ChunkingPolicy(
            mode=cast(ChunkingMode, mode),
            max_tokens=max_tokens,
            overlap_tokens=120,
            split_on_headings=split_on_headings,
        ),
        payload_schema=PayloadSchema(fields=dict(DEFAULT_PAYLOAD_FIELDS)),
        quality_checks=QualityChecks(max_tokens_per_chunk=context.hard_cap),
        test_queries=list(context.test_queries or []),
    )


class GPTPlanner:
    """Wrapper around the OpenAI API to request a plan."""

    def __init__(self, *, models: Sequence[str] | None = None, api_key: str) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package not available")
        self.client = OpenAI(api_key=api_key)
        ordered: list[str] = []
        seen: set[str] = set()
        for name in list(models or []) + list(ALLOWED_PLANNER_MODELS):
            if not name:
                continue
            if name not in seen:
                seen.add(name)
                ordered.append(name)
        self._models: list[str] = ordered or list(ALLOWED_PLANNER_MODELS)
        self._last_model_used: str | None = None

    @property
    def last_model_used(self) -> str | None:
        return self._last_model_used

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

        try:
            payload, model_used = call_llm_with_retries(
                client=self.client,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                logger=logger,
                candidates=self._models,
                temperature=0.0,
                max_output_tokens=2000,
            )
        except Exception as exc:
            raise RuntimeError(f"Planner call failed for all models: {exc}") from exc

        try:
            plan = self._build_plan_from_payload(
                payload,
                model_name=model_used,
                blocks=blocks,
                context=context,
            )
            self._last_model_used = model_used
            return plan
        except RuntimeError as exc:
            logger.debug(
                "GPT plan invalid for model %s, attempting repair: %s",
                model_used,
                exc,
            )
            repair_prompt = self._build_repair_prompt(schema, str(exc))
            retry_candidates = [model_used] + [
                name for name in self._models if name != model_used
            ]
            try:
                repaired_payload, repaired_model = call_llm_with_retries(
                    client=self.client,
                    system_prompt=system_prompt,
                    user_prompt=repair_prompt,
                    logger=logger,
                    candidates=retry_candidates,
                    temperature=0.0,
                    max_output_tokens=2000,
                )
            except Exception as repair_exc:
                raise RuntimeError(f"Plan repair failed: {repair_exc}") from repair_exc

            plan = self._build_plan_from_payload(
                repaired_payload,
                model_name=repaired_model,
                blocks=blocks,
                context=context,
            )
            self._last_model_used = repaired_model
            return plan

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

    def _build_plan_from_payload(
        self,
        payload: str,
        *,
        model_name: str,
        blocks: Sequence[Block],
        context: PlanningContext,
    ) -> VectorizationPlan:
        data = self._parse_plan_payload(payload)
        normalized = _normalize_plan_dict(
            data,
            blocks=blocks,
            context=context,
        )
        normalized["plan_source"] = "gpt(normalized)"
        normalized["planner_model"] = model_name
        normalized["collection_name"] = context.collection_name
        normalized["embedding_model"] = context.embedding_model
        normalized["distance"] = context.distance
        normalized["doc_type"] = context.file_type
        try:
            return VectorizationPlan.from_dict(normalized)
        except Exception as exc:
            raise RuntimeError(f"Plan validation failed: {exc}") from exc

    def _parse_plan_payload(self, payload: str) -> Dict[str, object]:
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON from GPT: {exc}") from exc


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
        candidate_models = self._candidate_models(context)
        fallback_key = "fallback-ru" if _looks_like_russian_code(blocks) else "fallback"
        for model_name in [*candidate_models, fallback_key]:
            cache_key = self._cache_key(blocks, context, model_name)
            cached = self._plan_cache.get(cache_key)
            if cached is not None:
                return VectorizationPlan.from_dict(json.loads(json.dumps(cached)))

        if context.api_key:
            try:
                planner = GPTPlanner(models=candidate_models, api_key=context.api_key)
            except Exception as exc:  # pragma: no cover - client initialisation issues
                logger.warning("Unable to initialise GPT planner: %s", exc)
            else:
                try:
                    plan = planner.plan(blocks=blocks, context=context)
                except Exception as exc:
                    logger.warning("GPT planning failed for all models: %s", exc)
                else:
                    model_key = planner.last_model_used or (
                        candidate_models[0] if candidate_models else "gpt"
                    )
                    cache_key = self._cache_key(blocks, context, model_key)
                    self._plan_cache[cache_key] = plan.model_dump()
                    return plan

        plan = build_fallback_plan(blocks=blocks, context=context)
        cache_key = self._cache_key(blocks, context, fallback_key)
        self._plan_cache[cache_key] = plan.model_dump()
        return plan

    def cache_info(self) -> Dict[str, int]:
        return {"entries": len(self._plan_cache)}

    def _cache_key(
        self, blocks: Sequence[Block], context: PlanningContext, model_name: str
    ) -> Tuple[str, str, str]:
        return (
            self._hash_blocks(blocks),
            context.file_type,
            model_name,
        )

    def _candidate_models(self, context: PlanningContext) -> list[str]:
        models: list[str] = []
        preferred = context.chat_model
        if preferred and preferred in ALLOWED_PLANNER_MODELS and preferred not in models:
            models.append(preferred)
        for model_name in ALLOWED_PLANNER_MODELS:
            if model_name not in models:
                models.append(model_name)
        return models

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
        if stats["max_tokens"] > plan.chunking_policy.max_tokens:
            raise RuntimeError(
                "Preview chunk tokens exceed configured max_tokens"
            )
        return PlanPreview(plan=plan, chunks=chunk_previews, stats=stats)


__all__ = [
    "PlanningContext",
    "VectorizationPlanner",
    "GPTPlanner",
    "build_fallback_plan",
]

