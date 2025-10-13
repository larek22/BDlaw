"""LLM-assisted ingestion router for determining document plans."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel, Field, ValidationError

try:  # pragma: no cover - optional dependency
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover - optional dependency not installed
    OpenAI = None  # type: ignore

from .data_repository import DataRepository
from .settings import AppSettings

logger = logging.getLogger(__name__)


class VectorHeadConfig(BaseModel):
    title_source: str = Field(default="heading")
    body_source: str = Field(default="all_text")


class SegmentationConfig(BaseModel):
    strategy: str = Field(default="regex")
    regexes: Dict[str, str] = Field(default_factory=dict)
    heading_levels: List[str] = Field(default_factory=list)


class MetadataField(BaseModel):
    name: str
    source: str
    pattern: Optional[str] = None


class PostFilterConfig(BaseModel):
    min_chars_per_chunk: Optional[int] = None
    drop_boilerplate: bool = False


class IngestionPlan(BaseModel):
    doc_type: str = Field(default="generic_text")
    languages: List[str] = Field(default_factory=list)
    chunk_tokens: int = Field(default=900)
    token_overlap: int = Field(default=120)
    vector_heads: VectorHeadConfig = Field(default_factory=VectorHeadConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    metadata_schema: List[MetadataField] = Field(default_factory=list)
    post_filters: Optional[PostFilterConfig] = None


_ROUTER_FUNCTION_SPEC = {
    "name": "propose_ingestion_plan",
    "description": "Return a precise ingestion plan for the document.",
    "parameters": {
        "type": "object",
        "properties": {
            "doc_type": {
                "type": "string",
                "enum": [
                    "legal_code",
                    "contract",
                    "regulation",
                    "case_law",
                    "book_epub",
                    "html_article",
                    "scanned_pdf",
                    "generic_text",
                    "table_like",
                ],
            },
            "languages": {
                "type": "array",
                "items": {"type": "string"},
            },
            "chunk_tokens": {"type": "integer", "minimum": 200, "maximum": 2000},
            "token_overlap": {"type": "integer", "minimum": 0, "maximum": 400},
            "vector_heads": {
                "type": "object",
                "properties": {
                    "title_source": {
                        "type": "string",
                        "enum": [
                            "heading",
                            "article_caption",
                            "first_line",
                            "filename",
                            "none",
                        ],
                    },
                    "body_source": {
                        "type": "string",
                        "enum": [
                            "section_body",
                            "article_body",
                            "paragraphs",
                            "all_text",
                        ],
                    },
                },
                "required": ["title_source", "body_source"],
            },
            "segmentation": {
                "type": "object",
                "properties": {
                    "strategy": {
                        "type": "string",
                        "enum": [
                            "regex",
                            "heading_outline",
                            "page_blocks",
                            "table_rows",
                        ],
                    },
                    "regexes": {
                        "type": "object",
                        "properties": {
                            "article": {"type": "string"},
                            "chapter": {"type": "string"},
                            "section": {"type": "string"},
                        },
                        "additionalProperties": False,
                    },
                    "heading_levels": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["strategy"],
            },
            "metadata_schema": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "source": {
                            "type": "string",
                            "enum": ["regex_group", "heading", "fixed", "heuristic"],
                        },
                        "pattern": {"type": "string"},
                    },
                    "required": ["name", "source"],
                },
            },
            "post_filters": {
                "type": "object",
                "properties": {
                    "min_chars_per_chunk": {"type": "integer", "minimum": 120},
                    "drop_boilerplate": {"type": "boolean"},
                },
            },
        },
        "required": ["doc_type", "chunk_tokens", "token_overlap", "vector_heads", "segmentation"],
    },
}


@dataclass
class RouterSniffInfo:
    path: Path
    sample_text: str
    headings: List[str]
    language: Optional[str]
    mime_type: Optional[str]


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class RouterCache:
    """Simple SQLite backed cache for ingestion plans."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS plans (
                    cache_key TEXT PRIMARY KEY,
                    plan_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def get(self, cache_key: str) -> Optional[IngestionPlan]:
        with sqlite3.connect(self.path) as conn:
            row = conn.execute(
                "SELECT plan_json FROM plans WHERE cache_key = ?", (cache_key,)
            ).fetchone()
        if not row:
            return None
        try:
            payload = json.loads(row[0])
            return IngestionPlan.model_validate(payload)
        except (ValidationError, json.JSONDecodeError) as exc:
            logger.warning("Cached ingestion plan invalid for key %s: %s", cache_key, exc)
            return None

    def set(self, cache_key: str, plan: IngestionPlan) -> None:
        payload = plan.model_dump()
        with sqlite3.connect(self.path) as conn:
            conn.execute(
                "REPLACE INTO plans(cache_key, plan_json) VALUES(?, ?)",
                (cache_key, json.dumps(payload, ensure_ascii=False)),
            )


class IngestionRouter:
    """Decide how a document should be split and annotated for ingestion."""

    def __init__(self, settings: AppSettings, repository: DataRepository) -> None:
        self.settings = settings
        self.repository = repository
        self.cache = RouterCache(repository.router_cache_path())
        self._client = None
        router_model = getattr(settings, "router", None)
        if router_model and router_model.use_llm and settings.openai_api_key and OpenAI is not None:
            try:
                self._client = OpenAI(api_key=settings.openai_api_key)
            except Exception as exc:  # pragma: no cover
                logger.warning("Failed to initialise OpenAI client for router: %s", exc)
                self._client = None

    def route(self, sniff: RouterSniffInfo) -> IngestionPlan:
        cache_key = self._cache_key(sniff)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        plan = self._generate_plan(sniff)
        self.cache.set(cache_key, plan)
        return plan

    def _cache_key(self, sniff: RouterSniffInfo) -> str:
        text_sample = sniff.sample_text.encode("utf-8")[:4096]
        headings_blob = "\n".join(sniff.headings).encode("utf-8")
        combined = b"|".join(
            [
                str(sniff.path).encode("utf-8"),
                text_sample,
                headings_blob,
                (sniff.language or "").encode("utf-8"),
                (sniff.mime_type or "").encode("utf-8"),
            ]
        )
        return _hash_bytes(combined)

    def _generate_plan(self, sniff: RouterSniffInfo) -> IngestionPlan:
        heuristics_plan = self._heuristic_plan(sniff)
        router_settings = getattr(self.settings, "router", None)
        if self._client is None or router_settings is None or not router_settings.use_llm:
            return heuristics_plan

        max_calls = max(1, router_settings.max_calls_per_document)
        attempts = 0
        while attempts < max_calls:
            attempts += 1
            try:
                payload = self._invoke_router(sniff)
                return IngestionPlan.model_validate(payload)
            except Exception as exc:  # pragma: no cover - LLM failures fallback
                logger.warning("Router LLM call failed (%s/%s): %s", attempts, max_calls, exc)
                if attempts >= max_calls:
                    break
        return heuristics_plan

    def _heuristic_plan(self, sniff: RouterSniffInfo) -> IngestionPlan:
        text = sniff.sample_text.lower()
        headings = "\n".join(sniff.headings).lower()
        if "статья" in text or "статья" in headings:
            return IngestionPlan(
                doc_type="legal_code",
                languages=[sniff.language or "ru"],
                chunk_tokens=900,
                token_overlap=120,
                vector_heads=VectorHeadConfig(
                    title_source="article_caption",
                    body_source="article_body",
                ),
                segmentation=SegmentationConfig(
                    strategy="regex",
                    regexes={
                        "article": r"(?m)^\s*Статья\s+(\d+(?:\.\d+)?)\s*(?:\.|:)?\s*(.*)$",
                        "chapter": r"(?m)^\s*Глава\s+(\d+)\s*(.*)$",
                        "section": r"(?m)^\s*Раздел\s+([IVXLC]+|\d+)\s*(.*)$",
                    },
                ),
                metadata_schema=[
                    MetadataField(
                        name="article", source="regex_group", pattern=r"Статья\s+(\d+(?:\.\d+)?)"
                    ),
                    MetadataField(
                        name="chapter", source="regex_group", pattern=r"Глава\s+(\d+)"
                    ),
                ],
                post_filters=PostFilterConfig(min_chars_per_chunk=200, drop_boilerplate=True),
            )
        return IngestionPlan(
            doc_type="generic_text",
            languages=[sniff.language or "ru"],
            chunk_tokens=800,
            token_overlap=80,
            vector_heads=VectorHeadConfig(title_source="first_line", body_source="paragraphs"),
            segmentation=SegmentationConfig(strategy="heading_outline", heading_levels=sniff.headings[:5]),
            post_filters=PostFilterConfig(min_chars_per_chunk=150, drop_boilerplate=False),
        )

    def _invoke_router(self, sniff: RouterSniffInfo) -> Dict[str, Any]:  # pragma: no cover - network path
        assert self._client is not None
        router_settings = getattr(self.settings, "router")
        sample = sniff.sample_text[: router_settings.sample_bytes]
        headings = sniff.headings[:10]
        payload = {
            "mime": sniff.mime_type or "unknown",
            "language": sniff.language or "unknown",
            "headings": headings,
            "sample_text": sample,
        }
        response = self._client.responses.create(
            model=router_settings.model,
            input=json.dumps(payload, ensure_ascii=False),
            functions=[_ROUTER_FUNCTION_SPEC],
            function_call={"name": "propose_ingestion_plan"},
        )
        choice = response.output[0]
        if getattr(choice, "type", "") != "function_call":
            raise RuntimeError("Router response missing function call result")
        arguments = getattr(choice, "function_call", {}).get("arguments")
        if not arguments:
            raise RuntimeError("Router response missing arguments")
        return json.loads(arguments)


def build_sniff_info(path: Path, pages: Iterable[str], *, mime_type: str | None = None) -> RouterSniffInfo:
    sample_parts: List[str] = []
    headings: List[str] = []
    language_guess = None
    for idx, page in enumerate(pages):
        if idx == 0:
            sample_parts.append(page[:2000])
        elif idx == 1:
            sample_parts.append(page[:1500])
        elif idx == 5:
            sample_parts.append(page[:1500])
        for line in page.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if re_heading := re_heading_match(stripped):
                headings.append(re_heading)
        if language_guess is None and any("статья" in part.lower() for part in page.splitlines()[:10]):
            language_guess = "ru"
    sample_text = "\n".join(sample_parts)[:6000]
    return RouterSniffInfo(
        path=path,
        sample_text=sample_text,
        headings=headings,
        language=language_guess,
        mime_type=mime_type,
    )


def re_heading_match(line: str) -> Optional[str]:
    lowered = line.lower()
    if lowered.startswith("статья") or lowered.startswith("глава") or lowered.startswith("раздел"):
        return line
    return None


__all__ = [
    "IngestionPlan",
    "RouterSniffInfo",
    "IngestionRouter",
    "build_sniff_info",
]

