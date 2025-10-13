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
    prefer_grpc: bool = Field(default_factory=lambda: _env_bool("QDRANT_PREFER_GRPC", True))
    grpc_port: int = Field(default_factory=lambda: _env_int("QDRANT_GRPC_PORT", 6334))
    connect_timeout_seconds: float = Field(default_factory=lambda: _env_float("QDRANT_CONNECT_TIMEOUT", 10.0))
    read_timeout_seconds: float = Field(default_factory=lambda: _env_float("QDRANT_READ_TIMEOUT", 60.0))
    write_timeout_seconds: float = Field(default_factory=lambda: _env_float("QDRANT_WRITE_TIMEOUT", 240.0))
    target_batch_bytes: int = Field(default_factory=lambda: _env_int("INGEST_TARGET_BATCH_BYTES", 2_000_000))
    max_retries: int = Field(default_factory=lambda: max(1, _env_int("QDRANT_MAX_RETRIES", 5)))
    retry_initial_delay: float = Field(default_factory=lambda: _env_float("QDRANT_RETRY_INITIAL_DELAY", 1.0))
    retry_max_delay: float = Field(default_factory=lambda: _env_float("QDRANT_RETRY_MAX_DELAY", 15.0))
    upsert_wait: bool = Field(default_factory=lambda: _env_bool("QDRANT_UPSERT_WAIT", False))
    upsert_ordering: str = Field(default_factory=lambda: (_env_str("QDRANT_UPSERT_ORDERING", "weak") or "weak"))
    title_max_chars: int = Field(default_factory=lambda: _env_int("TEXT_TITLE_MAX", 256))
    preview_max_chars: int = Field(default_factory=lambda: _env_int("TEXT_PREVIEW_MAX", 1200))
    max_batch_size: int = Field(default_factory=lambda: max(1, _env_int("QDRANT_MAX_BATCH_SIZE", 128)))
    min_batch_size: int = Field(default_factory=lambda: max(1, _env_int("QDRANT_MIN_BATCH_SIZE", 16)))
    use_wait: bool = Field(default_factory=lambda: _env_bool("QDRANT_USE_WAIT", False))
    allow_insecure_https_without_api_key: bool = Field(
        default_factory=lambda: _env_bool("QDRANT_ALLOW_INSECURE_HTTPS", False)
    )

    model_config = {"extra": "ignore", "validate_assignment": True}


class IngestSettings(BaseModel):
    chunk_size_chars: int = Field(default_factory=lambda: _env_int("CHUNK_SIZE", 1200))
    chunk_overlap_chars: int = Field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 120))
    chunk_size_tokens: int = Field(default_factory=lambda: _env_int("CHUNK_SIZE_TOKENS", 1200))
    overlap_tokens: int = Field(default_factory=lambda: _env_int("CHUNK_OVERLAP_TOKENS", 120))
    validation_sample_k: int = Field(default_factory=lambda: max(1, _env_int("INGEST_VALIDATION_SAMPLE_K", 8)))
    atomic_alias_swap: bool = Field(default_factory=lambda: _env_bool("INGEST_ATOMIC_ALIAS_SWAP", True))
    dry_run: bool = Field(default_factory=lambda: _env_bool("INGEST_DRY_RUN", False))
    enable_new_loaders: bool = Field(default_factory=lambda: _env_bool("INGEST_ENABLE_NEW_LOADERS", False))
    force_reingest_if_empty: bool = Field(
        default_factory=lambda: _env_bool("INGEST_FORCE_REINGEST_IF_EMPTY", True)
    )

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
    reranker_mode: str = Field(default_factory=lambda: _env_str("RERANKER_MODE", "off") or "off")
    reranker_weight: float = Field(default_factory=lambda: _env_float("RERANKER_WEIGHT", 0.7))
    reranker_model: str = Field(default_factory=lambda: _env_str("RERANKER_MODEL", "BAAI/bge-reranker-large") or "BAAI/bge-reranker-large")
    reranker_cache_ttl_seconds: int = Field(default_factory=lambda: _env_int("RERANKER_CACHE_TTL", 300))
    reranker_batch_size: int = Field(default_factory=lambda: max(1, _env_int("RERANKER_BATCH_SIZE", 8)))
    reranker_top_n: int = Field(default_factory=lambda: max(1, _env_int("RERANKER_TOP_N", 12)))
    enable_law_filters: bool = Field(default_factory=lambda: _env_bool("SEARCH_ENABLE_LAW_FILTERS", True))
    part_no: int | None = Field(default=None)
    chapter_no: int | None = Field(default=None)
    article_no_int: int | None = Field(default=None)

    model_config = {"extra": "ignore", "validate_assignment": True}


class PlannerSettings(BaseModel):
    use_llm: bool = Field(default_factory=lambda: _env_bool("PLANNER_USE_LLM", True))
    temperature: float = Field(default_factory=lambda: _env_float("PLANNER_TEMPERATURE", 0.0))
    max_tokens: int = Field(default_factory=lambda: _env_int("PLANNER_MAX_TOKENS", 700))
    prompt_version: str = Field(default_factory=lambda: _env_str("PLANNER_PROMPT_VERSION", "v1.0") or "v1.0")
    sample_bytes: int = Field(default_factory=lambda: _env_int("PLANNER_SAMPLE_BYTES", 20000))
    gpt41_budget_tokens_per_file: int = Field(
        default_factory=lambda: _env_int("PLANNER_GPT41_BUDGET_TOKENS", 60_000)
    )
    enable_cache: bool = Field(default_factory=lambda: _env_bool("PLANNER_ENABLE_CACHE", True))

    model_config = {"extra": "ignore", "validate_assignment": True}


class ChunkingSettings(BaseModel):
    max_tokens: int = Field(default_factory=lambda: _env_int("CHUNK_MAX_TOKENS", 900))
    overlap_tokens: int = Field(default_factory=lambda: _env_int("CHUNK_OVERLAP_TOKENS", 120))
    tokenizer: str = Field(default_factory=lambda: _env_str("EMBED_TOKENIZER", "cl100k_base") or "cl100k_base")

    model_config = {"extra": "ignore", "validate_assignment": True}


class OcrSettings(BaseModel):
    enabled: bool = Field(default_factory=lambda: _env_bool("OCR_ENABLED", True))
    enable: bool = Field(default_factory=lambda: _env_bool("OCR_ENABLE", False))
    min_text_chars: int = Field(default_factory=lambda: _env_int("OCR_MIN_TEXT_CHARS", 800))
    min_average_chars_per_line: float = Field(
        default_factory=lambda: float(_env_float("OCR_MIN_AVG_CHARS", 5.0))
    )
    timeout_seconds: int = Field(default_factory=lambda: _env_int("OCR_TIMEOUT_SECONDS", 180))
    languages: str = Field(default_factory=lambda: _env_str("OCR_LANGUAGES", "rus+eng") or "rus+eng")
    max_retries: int = Field(default_factory=lambda: max(1, _env_int("OCR_MAX_RETRIES", 2)))

    model_config = {"extra": "ignore", "validate_assignment": True}


class RouterSettings(BaseModel):
    use_llm: bool = Field(default_factory=lambda: _env_bool("ROUTER_USE_LLM", True))
    model: str = Field(default_factory=lambda: _env_str("ROUTER_MODEL", "gpt-4.1-mini") or "gpt-4.1-mini")
    sample_bytes: int = Field(default_factory=lambda: _env_int("ROUTER_SAMPLE_BYTES", 4096))
    max_calls_per_document: int = Field(default_factory=lambda: _env_int("ROUTER_MAX_CALLS", 2))

    model_config = {"extra": "ignore", "validate_assignment": True}


class PayloadSettings(BaseModel):
    slim_body_text: bool = Field(
        default_factory=lambda: _env_bool("PAYLOAD_SLIM_BODY_TEXT", False)
    )

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
    ocr: OcrSettings = Field(default_factory=OcrSettings)
    router: RouterSettings = Field(default_factory=RouterSettings)
    payload: PayloadSettings = Field(default_factory=PayloadSettings)
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
    "OcrSettings",
    "RouterSettings",
    "PayloadSettings",
    "VerificationSettings",
    "CONFIG_PATH",
]
