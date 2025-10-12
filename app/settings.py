from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional


@dataclass(frozen=True)
class QdrantConfig:
    """Minimal runtime configuration for Qdrant connectivity."""

    url: str = os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key: Optional[str] = os.environ.get("QDRANT_API_KEY")
    prefer_grpc: bool = os.environ.get("QDRANT_USE_GRPC", "true").lower() == "true"
    grpc_port: int = int(os.environ.get("QDRANT_GRPC_PORT", "6334"))
    tls: bool = os.environ.get("QDRANT_TLS", "true").lower() == "true"
    connect_timeout: float = float(os.environ.get("QDRANT_CONNECT_TIMEOUT", "30"))
    read_timeout: float = float(os.environ.get("QDRANT_READ_TIMEOUT", "120"))
    write_timeout: float = float(os.environ.get("QDRANT_WRITE_TIMEOUT", "120"))
    target_batch_bytes: int = int(os.environ.get("INGEST_TARGET_BATCH_BYTES", "800000"))
    max_points_per_batch: int = int(os.environ.get("BATCH_POINTS_MAX", "32"))
    max_retries: int = int(os.environ.get("INGEST_MAX_RETRIES", "5"))
    alias_name: str = os.environ.get("QDRANT_ALIAS", "GK_RF2")
    collection_prefix: str = os.environ.get("QDRANT_COLLECTION_PREFIX", "GK_RF2_d3072")
    wait_for_upsert: bool = os.environ.get("UPSERT_WAIT", "false").lower() == "true"
    ordering: str = os.environ.get("UPSERT_ORDERING", "weak")


@dataclass(frozen=True)
class PayloadLimits:
    title_max_chars: int = int(os.environ.get("TEXT_TITLE_MAX", "256"))
    preview_max_chars: int = int(os.environ.get("TEXT_PREVIEW_MAX", "1200"))


@dataclass(frozen=True)
class IngestRuntime:
    qdrant: QdrantConfig = QdrantConfig()
    payload_limits: PayloadLimits = PayloadLimits()


__all__ = ["IngestRuntime", "QdrantConfig", "PayloadLimits"]
