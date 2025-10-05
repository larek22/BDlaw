"""Configuration handling for BDlaw."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml


DEFAULT_CONFIG_PATH = Path("config.yaml")


@dataclass
class OCRConfig:
    languages: List[str] = field(default_factory=lambda: ["rus", "eng"])
    tesseract_path: Optional[str] = None


@dataclass
class ChunkingConfig:
    max_words: int = 750
    overlap_words: int = 80


@dataclass
class NormalizationConfig:
    replace_yo: bool = False


@dataclass
class EmbeddingConfig:
    model: str = "text-embedding-3-small"
    batch_size: int = 128


@dataclass
class QdrantConfig:
    host: str = "localhost"
    port: int = 6333
    collection: str = "laws"
    prefer_grpc: bool = False


@dataclass
class LLMConfig:
    answer_model: str = "gpt-4.1-mini"
    judge_model: str = "gpt-4.1-mini"


@dataclass
class AppConfig:
    ocr: OCRConfig = field(default_factory=OCRConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    locale: str = "ru"
    theme: str = "dark"


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    """Load configuration from *path* or return defaults."""

    if not path.exists():
        return AppConfig()
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return AppConfig(
        ocr=OCRConfig(**data.get("ocr", {})),
        normalization=NormalizationConfig(**data.get("normalization", {})),
        chunking=ChunkingConfig(**data.get("chunking", {})),
        embedding=EmbeddingConfig(**data.get("embedding", {})),
        qdrant=QdrantConfig(**data.get("qdrant", {})),
        llm=LLMConfig(**data.get("llm", {})),
        locale=data.get("locale", "ru"),
        theme=data.get("theme", "dark"),
    )


__all__ = [
    "AppConfig",
    "ChunkingConfig",
    "EmbeddingConfig",
    "LLMConfig",
    "NormalizationConfig",
    "OCRConfig",
    "QdrantConfig",
    "load_config",
]
