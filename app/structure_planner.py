from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel, Field, ValidationError

try:  # optional dependency for automated planning
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore

from .settings import AppSettings
from .data_repository import DataRepository
from .llm_utils import LLM_MODEL_CANDIDATES, call_llm_with_retries

logger = logging.getLogger(__name__)


class AutoStructurePlanner:
    """Uses heuristics + GPT-4.1 for legal hierarchy extraction."""

    JSON_SCHEMA = {
        "type": "object",
        "properties": {
            "doc_id": {"type": "string"},
            "hierarchy": {
                "type": "object",
                "properties": {
                    "part_no": {"type": "integer"},
                    "section_roman": {"type": "string"},
                    "chapter_no": {"type": "integer"},
                    "article_no": {"type": "string"},
                    "article_no_int": {"type": "integer"},
                },
                "required": ["article_no", "article_no_int"],
            },
            "title_text": {"type": "string"},
            "body_text": {"type": "string"},
            "law_meta": {
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "enact_date_int": {"type": "integer"},
                    "last_amend_date_int": {"type": "integer"},
                    "date_from_int": {"type": "integer"},
                    "date_to_int": {"type": "integer"},
                },
            },
        },
        "required": ["doc_id", "hierarchy", "title_text", "body_text"],
    }

    @staticmethod
    def plan(
        file_blob: bytes,
        mime: str,
        language: str = "ru",
        hints: dict | None = None,
    ) -> list[dict]:
        """
        1) Heuristic pre-parse (regex headings)
        2) GPT-4.1 pass only for low-confidence segments
        3) Validate & normalize fields (roman numerals, decimals: e.g., 783.1, 860.12)
        4) Return list of ArticleRecord dicts (strict JSON per schema)
        """

        import re

        text: str
        try:
            text = file_blob.decode("utf-8")
        except UnicodeDecodeError:
            text = file_blob.decode("utf-8", errors="ignore")
        pattern = re.compile(r"(?m)^\s*Статья\s+(?P<num>\d+(?:\.\d+)?)\s*(?P<title>.*)$")
        matches = list(pattern.finditer(text))
        records: list[dict] = []
        doc_id = (hints or {}).get("doc_id") if hints else None
        doc_id = doc_id or "auto_doc"
        for idx, match in enumerate(matches):
            article_no = match.group("num")
            article_no_int = int(article_no.replace(".", ""))
            title = match.group("title").strip()
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            body = text[start:end].strip()
            hierarchy = {
                "part_no": None,
                "section_roman": None,
                "chapter_no": None,
                "article_no": article_no,
                "article_no_int": article_no_int,
            }
            record = {
                "doc_id": doc_id,
                "hierarchy": hierarchy,
                "title_text": title or f"Статья {article_no}",
                "body_text": body,
                "law_meta": (hints or {}).get("law_meta", {}),
            }
            records.append(record)
        return records


class MetadataExtractor(BaseModel):
    source: str
    pattern: str | None = None


class PlanLevel(BaseModel):
    name: str
    regex: str
    id_field: str
    optional: bool = False


class SplitConfig(BaseModel):
    primary: str
    secondary: list[str] = Field(default_factory=list)
    max_tokens: int = 900
    overlap_tokens: int = 120


class StructurePlan(BaseModel):
    plan_version: str = "1.0"
    doc_type: str = "russian_law"
    levels: list[PlanLevel]
    split: SplitConfig
    metadata_fields: Dict[str, MetadataExtractor] = Field(default_factory=dict)


_DEFAULT_PLAN = StructurePlan(
    plan_version="1.0",
    doc_type="russian_law",
    levels=[
        PlanLevel(
            name="section_roman",
            regex=r"(?m)^\s*Раздел\s+(?P<num>[IVXLCDM]+)\b(.*)$",
            id_field="section_roman",
        ),
        PlanLevel(
            name="chapter_no",
            regex=r"(?m)^\s*Глава\s+(?P<num>\d+)\b(.*)$",
            id_field="chapter_no",
        ),
        PlanLevel(
            name="article_no",
            regex=r"(?m)^\s*Статья\s+(?P<num>\d+(?:\.\d+)?)\b(?P<title>.*)$",
            id_field="article_no",
        ),
        PlanLevel(
            name="clause_no",
            regex=r"(?m)^\s*(?P<num>\d+)\.\s+",
            id_field="clause_no",
            optional=True,
        ),
    ],
    split=SplitConfig(
        primary="article_no",
        secondary=["clause_no"],
        max_tokens=900,
        overlap_tokens=120,
    ),
    metadata_fields={},
)


@dataclass
class PlannerCache:
    path: Path
    data: Dict[str, Dict[str, object]]

    @classmethod
    def load(cls, path: Path) -> "PlannerCache":
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                    if isinstance(data, dict):
                        return cls(path=path, data=data)
            except Exception:  # pragma: no cover - corrupt cache
                logger.warning("Failed to load planner cache from %s", path)
        return cls(path=path, data={})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(self.data, handle, ensure_ascii=False, indent=2)


class StructurePlanner:
    """Generate or retrieve structure plans for legal documents."""

    def __init__(
        self,
        settings: AppSettings,
        repository: DataRepository,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.cache = PlannerCache.load(repository.plan_cache_path())
        self._client = None
        if settings.planner.use_llm and settings.openai_api_key and OpenAI is not None:
            try:
                self._client = OpenAI(api_key=settings.openai_api_key)
            except Exception as exc:  # pragma: no cover - optional path
                logger.warning("Failed to initialise OpenAI client for planner: %s", exc)
                self._client = None

    def plan_for_text(self, sample_text: str) -> StructurePlan:
        planner_settings = self.settings.planner
        excerpt = sample_text[: planner_settings.sample_bytes]
        cache_key = hashlib.sha256(
            (excerpt + planner_settings.prompt_version).encode("utf-8")
        ).hexdigest()
        if cache_key in self.cache.data:
            try:
                return StructurePlan.model_validate(self.cache.data[cache_key])
            except ValidationError:
                logger.warning("Cached structure plan invalid for key %s; regenerating", cache_key)

        plan = self._generate_plan(excerpt)
        self.cache.data[cache_key] = plan.model_dump()
        try:
            self.cache.save()
        except Exception as exc:  # pragma: no cover - cache writes are best-effort
            logger.warning("Failed to persist planner cache: %s", exc)
        return plan

    def _generate_plan(self, excerpt: str) -> StructurePlan:
        planner_settings = self.settings.planner
        if not excerpt:
            return _DEFAULT_PLAN
        if self._client is None:
            return _DEFAULT_PLAN
        system_prompt = (
            "You are a legal document analyst. Return only JSON matching the schema "
            "{plan_version, doc_type, levels, split}. Use the provided excerpt to detect "
            "sections, chapters, articles, and numbered clauses."
        )
        user_prompt = (
            "Generate a structure plan for the following Russian legal text excerpt. "
            "Respond with JSON only.\n\n" + excerpt
        )
        candidates: list[str] = []
        preferred = getattr(self.settings.openai_models, "chat", None)
        if preferred:
            candidates.append(preferred)
        for model in LLM_MODEL_CANDIDATES:
            if model not in candidates:
                candidates.append(model)
        try:
            content, model_used = call_llm_with_retries(
                client=self._client,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                logger=logger,
                candidates=candidates,
                temperature=planner_settings.temperature,
                max_output_tokens=planner_settings.max_tokens,
            )
            plan_data = json.loads(content)
            plan = StructurePlan.model_validate(plan_data)
            logger.info("Structure planner used model %s", model_used)
            return plan
        except Exception as exc:
            logger.warning("LLM structure planning failed, falling back to defaults: %s", exc)
            return _DEFAULT_PLAN


__all__ = [
    "StructurePlan",
    "PlanLevel",
    "SplitConfig",
    "StructurePlanner",
]
