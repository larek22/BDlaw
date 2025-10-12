from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

CONFIG_DIR = Path(os.getenv("VECTOR_KB_HOME", Path.home() / ".vector_kb"))
CONFIG_PATH = CONFIG_DIR / "config.json"


def _env_str(key: str, default: str | None = None) -> str | None:
    value = os.getenv(key)
    if value is not None:
        return value
    return default


def _env_int(key: str, default: int) -> int:
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


class ModelSettings(BaseModel):
    embedding: str = Field(default_factory=lambda: _env_str("EMBEDDING_MODEL", "text-embedding-3-large") or "text-embedding-3-large")
    chat: str = Field(default_factory=lambda: _env_str("CHAT_MODEL", "gpt-4o-mini") or "gpt-4o-mini")

    model_config = {"extra": "ignore", "validate_assignment": True}


class QdrantSettings(BaseModel):
    url: str = Field(default_factory=lambda: _env_str("QDRANT_URL", "http://localhost:6333") or "http://localhost:6333")
    api_key: str | None = Field(default_factory=lambda: _env_str("QDRANT_API_KEY"))
    collection: str = Field(default_factory=lambda: _env_str("QDRANT_COLLECTION", "kb_docs_active") or "kb_docs_active")
    upsert_batch_size: int = Field(default_factory=lambda: _env_int("QDRANT_UPSERT_BATCH_SIZE", 128))
    timeout_seconds: float = Field(default_factory=lambda: _env_float("QDRANT_TIMEOUT_SECONDS", 30.0))

    model_config = {"extra": "ignore", "validate_assignment": True}


class IngestSettings(BaseModel):
    chunk_size_chars: int = Field(default_factory=lambda: _env_int("CHUNK_SIZE", 1200))
    chunk_overlap_chars: int = Field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 120))

    model_config = {"extra": "ignore", "validate_assignment": True}


class QuerySettings(BaseModel):
    top_k: int = Field(default_factory=lambda: _env_int("QUERY_TOP_K", 10))
    prefilter_limit: int = Field(default_factory=lambda: _env_int("QUERY_PREFILTER_LIMIT", 400))
    fusion_weight_title: float = Field(default_factory=lambda: _env_float("FUSION_WEIGHT_TITLE", 0.4))
    fusion_weight_body: float = Field(default_factory=lambda: _env_float("FUSION_WEIGHT_BODY", 0.6))
    fusion_rrf_k: int = Field(default_factory=lambda: _env_int("FUSION_RRF_K", 60))
    keyword_boost_weight: float = Field(default_factory=lambda: _env_float("QUERY_KEYWORD_BOOST_WEIGHT", 0.0))
    as_of_date: str | None = Field(default_factory=lambda: _env_str("QUERY_AS_OF_DATE"))
    as_of_start_date: str | None = Field(default_factory=lambda: _env_str("QUERY_AS_OF_START"))
    status_filter: str = Field(default_factory=lambda: _env_str("QUERY_STATUS", "active") or "active")

    model_config = {"extra": "ignore", "validate_assignment": True}


class PlannerSettings(BaseModel):
    use_llm: bool = Field(default_factory=lambda: _env_bool("PLANNER_USE_LLM", True))
    temperature: float = Field(default_factory=lambda: _env_float("PLANNER_TEMPERATURE", 0.0))
    max_tokens: int = Field(default_factory=lambda: _env_int("PLANNER_MAX_TOKENS", 700))
    prompt_version: str = Field(default_factory=lambda: _env_str("PLANNER_PROMPT_VERSION", "v1.0") or "v1.0")
    sample_bytes: int = Field(default_factory=lambda: _env_int("PLANNER_SAMPLE_BYTES", 20000))

    model_config = {"extra": "ignore", "validate_assignment": True}


class ChunkingSettings(BaseModel):
    max_tokens: int = Field(default_factory=lambda: _env_int("CHUNK_MAX_TOKENS", 900))
    overlap_tokens: int = Field(default_factory=lambda: _env_int("CHUNK_OVERLAP_TOKENS", 120))
    tokenizer: str = Field(default_factory=lambda: _env_str("EMBED_TOKENIZER", "cl100k_base") or "cl100k_base")

    model_config = {"extra": "ignore", "validate_assignment": True}


class VerificationSettings(BaseModel):
    min_top1_score: float = Field(default_factory=lambda: _env_float("VERIFICATION_MIN_TOP1_SCORE", 0.25))

    model_config = {"extra": "ignore", "validate_assignment": True}


class AppSettings(BaseModel):
    openai_api_key: str | None = Field(default_factory=lambda: _env_str("OPENAI_API_KEY"))
    openai_models: ModelSettings = Field(default_factory=ModelSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    ingest: IngestSettings = Field(default_factory=IngestSettings)
    query: QuerySettings = Field(default_factory=QuerySettings)
    planner: PlannerSettings = Field(default_factory=PlannerSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    verification: VerificationSettings = Field(default_factory=VerificationSettings)
    allow_destructive_migrations: bool = Field(default_factory=lambda: _env_bool("ALLOW_DESTRUCTIVE", False))

    model_config = {"extra": "ignore", "validate_assignment": True}

    def sanitised_dump(self) -> dict[str, Any]:
        data = self.model_dump()
        data["openai_api_key"] = ""
        qdrant_data = data.get("qdrant")
        if isinstance(qdrant_data, dict):
            qdrant_data["api_key"] = ""
        return data

    @classmethod
    def load(cls) -> "AppSettings":
        data: dict[str, Any] = {}
        if CONFIG_PATH.exists():
            try:
                with CONFIG_PATH.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except Exception:
                data = {}
        settings = cls(**data)
        # Always refresh secrets from environment variables
        settings.openai_api_key = _env_str("OPENAI_API_KEY")
        settings.qdrant.api_key = _env_str("QDRANT_API_KEY")
        settings.allow_destructive_migrations = _env_bool(
            "ALLOW_DESTRUCTIVE", settings.allow_destructive_migrations
        )
        return settings

    def save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        payload = self.sanitised_dump()
        with CONFIG_PATH.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


__all__ = [
    "AppSettings",
    "ModelSettings",
    "QdrantSettings",
    "IngestSettings",
    "QuerySettings",
    "PlannerSettings",
    "ChunkingSettings",
    "VerificationSettings",
    "CONFIG_PATH",
]
