"""Vectorization planning and chunking utilities."""
from .chunker import apply_plan, Chunk, ChunkerResult
from .models import (
    ChunkPreview,
    ChunkingMode,
    ChunkingPolicy,
    DistanceMetric,
    PlanPreview,
    PlanSource,
    QualityChecks,
    VectorizationPlan,
)
from .normalizer import Block, normalize_document
from .planner import PlanningContext, VectorizationPlanner, build_fallback_plan
from .service import VectorizationArtifacts, VectorizationService

__all__ = [
    "apply_plan",
    "Chunk",
    "ChunkerResult",
    "ChunkPreview",
    "ChunkingMode",
    "ChunkingPolicy",
    "DistanceMetric",
    "PlanPreview",
    "PlanSource",
    "QualityChecks",
    "VectorizationPlan",
    "Block",
    "normalize_document",
    "PlanningContext",
    "VectorizationPlanner",
    "build_fallback_plan",
    "VectorizationArtifacts",
    "VectorizationService",
]
