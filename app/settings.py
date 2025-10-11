from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict


CONFIG_DIR = Path(os.getenv("VECTOR_KB_HOME", Path.home() / ".vector_kb"))
CONFIG_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_PATH = CONFIG_DIR / "config.json"


def _env_default(key: str, default: str | None = None) -> str | None:
    value = os.getenv(key)
    if value:
        return value
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


def _env_int(key: str, default: int) -> int:
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass
class ModelSettings:
    embedding: str = "text-embedding-3-large"
    chat: str = "gpt-4o-mini"


@dataclass
class QdrantSettings:
    url: str = _env_default("QDRANT_URL", "http://localhost:6333") or "http://localhost:6333"
    api_key: str = _env_default("QDRANT_API_KEY", "") or ""
    collection: str = _env_default("QDRANT_COLLECTION", "kb_docs_v1") or "kb_docs_v1"
    upsert_batch_size: int = _env_int("QDRANT_UPSERT_BATCH_SIZE", 128)
    timeout_seconds: float = _env_float("QDRANT_TIMEOUT_SECONDS", 30.0)


@dataclass
class IngestSettings:
    chunk_size_chars: int = _env_int("CHUNK_SIZE", 1200)
    chunk_overlap_chars: int = _env_int("CHUNK_OVERLAP", 120)


@dataclass
class QuerySettings:
    top_k: int = _env_int("QUERY_TOP_K", 10)
    prefilter_limit: int = _env_int("QUERY_PREFILTER_LIMIT", 400)
    fusion_weight_title: float = _env_float("FUSION_WEIGHT_TITLE", 0.4)
    fusion_weight_body: float = _env_float("FUSION_WEIGHT_BODY", 0.6)
    fusion_rrf_k: int = _env_int("FUSION_RRF_K", 60)
    as_of_date: str | None = _env_default("QUERY_AS_OF_DATE", None)
    as_of_start_date: str | None = _env_default("QUERY_AS_OF_START", None)
    status_filter: str = _env_default("QUERY_STATUS", "active") or "active"


@dataclass
class PlannerSettings:
    use_llm: bool = _env_bool("PLANNER_USE_LLM", True)
    temperature: float = _env_float("PLANNER_TEMPERATURE", 0.0)
    max_tokens: int = _env_int("PLANNER_MAX_TOKENS", 700)
    prompt_version: str = _env_default("PLANNER_PROMPT_VERSION", "v1.0") or "v1.0"
    sample_bytes: int = _env_int("PLANNER_SAMPLE_BYTES", 20000)


@dataclass
class ChunkingSettings:
    max_tokens: int = _env_int("CHUNK_MAX_TOKENS", 900)
    overlap_tokens: int = _env_int("CHUNK_OVERLAP_TOKENS", 120)
    tokenizer: str = _env_default("EMBED_TOKENIZER", "cl100k_base") or "cl100k_base"


@dataclass
class VerificationSettings:
    min_top1_score: float = _env_float("VERIFICATION_MIN_TOP1_SCORE", 0.25)


@dataclass
class AppSettings:
    openai_api_key: str = field(default_factory=lambda: _env_default("OPENAI_API_KEY", ""))
    openai_models: ModelSettings = field(default_factory=ModelSettings)
    qdrant: QdrantSettings = field(default_factory=QdrantSettings)
    ingest: IngestSettings = field(default_factory=IngestSettings)
    query: QuerySettings = field(default_factory=QuerySettings)
    planner: PlannerSettings = field(default_factory=PlannerSettings)
    chunking: ChunkingSettings = field(default_factory=ChunkingSettings)
    verification: VerificationSettings = field(default_factory=VerificationSettings)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppSettings":
        return cls(
            openai_api_key=data.get("openai_api_key", _env_default("OPENAI_API_KEY", "")),
            openai_models=ModelSettings(**data.get("openai_models", {})),
            qdrant=QdrantSettings(**data.get("qdrant", {})),
            ingest=IngestSettings(**data.get("ingest", {})),
            query=QuerySettings(**data.get("query", {})),
            planner=PlannerSettings(**data.get("planner", {})),
            chunking=ChunkingSettings(**data.get("chunking", {})),
            verification=VerificationSettings(**data.get("verification", {})),
        )

    @classmethod
    def load(cls) -> "AppSettings":
        if CONFIG_PATH.exists():
            with CONFIG_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
                return cls.from_dict(data)
        return cls()

    def save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with CONFIG_PATH.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


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
