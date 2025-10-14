"""Vectorization planning and chunking utilities."""
from .auto_pipeline import (
    AutoChunk,
    AutoPipelineConfig,
    AutoPipelineDeps,
    AutoVectorizationPipeline,
    FileProcessingResult,
)
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
from .quality import QualityMetrics, QualityReport, run_quality_checks
from .service import VectorizationArtifacts, VectorizationService

__all__ = [
    "AutoChunk",
    "AutoPipelineConfig",
    "AutoPipelineDeps",
    "AutoVectorizationPipeline",
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
    "QualityMetrics",
    "QualityReport",
    "run_quality_checks",
    "VectorizationArtifacts",
    "VectorizationService",
    "FileProcessingResult",
]
