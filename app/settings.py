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


@dataclass
class ModelSettings:
    embedding: str = "text-embedding-3-large"
    chat: str = "gpt-4o-mini"


@dataclass
class QdrantSettings:
    url: str = "http://localhost:6333"
    api_key: str = ""
    collection: str = "kb_docs_v1"
    upsert_batch_size: int = 128
    timeout_seconds: float = 30.0


@dataclass
class IngestSettings:
    chunk_size_chars: int = 1200
    chunk_overlap_chars: int = 120


@dataclass
class QuerySettings:
    top_k: int = 10


@dataclass
class AppSettings:
    openai_api_key: str = field(default_factory=lambda: _env_default("OPENAI_API_KEY", ""))
    openai_models: ModelSettings = field(default_factory=ModelSettings)
    qdrant: QdrantSettings = field(default_factory=QdrantSettings)
    ingest: IngestSettings = field(default_factory=IngestSettings)
    query: QuerySettings = field(default_factory=QuerySettings)

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
    "CONFIG_PATH",
]
