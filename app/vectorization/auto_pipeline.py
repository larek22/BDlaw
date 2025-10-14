"""Hands-off auto vectorization and ingestion pipeline."""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Sequence

from .chunker import ChunkerResult, apply_plan
from .models import DistanceMetric, VectorizationPlan
from .normalizer import normalize_document
from .planner import PlanningContext, VectorizationPlanner, build_fallback_plan
from .quality import QualityMetrics, QualityReport, run_quality_checks
from .tokenization import DEFAULT_TOKENIZER, Tokenizer

logger = logging.getLogger(__name__)

RunMode = str  # "auto" | "review" | "manual"


@dataclass(slots=True)
class AutoChunk:
    chunk_id: str
    text: str
    tokens: int
    start: int
    end: int
    meta: Dict[str, object]


@dataclass(slots=True)
class AutoPipelineConfig:
    collection_name: str
    embedding_model: str = "text-embedding-3-large"
    distance: DistanceMetric = "cosine"
    hard_cap: int = 1200
    run_mode: RunMode = "auto"
    coverage_threshold: float = 0.95
    output_dir: Path = Path("data/manifests")
    preview_limit: int = 5
    manual_plan: VectorizationPlan | None = None
    test_queries: Sequence[str] | None = None


@dataclass(slots=True)
class AutoPipelineDeps:
    embed_batch: Callable[[Sequence[str], str], Sequence[Sequence[float]]] | None = None
    upsert_batch: Callable[[Sequence[AutoChunk], Sequence[Sequence[float]], VectorizationPlan, str], None] | None = None
    alias_swap: Callable[[str], None] | None = None


@dataclass(slots=True)
class FileProcessingResult:
    path: Path
    plan: VectorizationPlan
    plan_hash: str
    quality: QualityReport
    ingested: bool
    escalated: bool
    alias_swapped: bool
    preview: List[Dict[str, object]]
    plan_path: Path | None
    preview_path: Path | None
    metrics_path: Path | None
    status: str
    notes: List[str] = field(default_factory=list)


class AutoVectorizationPipeline:
    def __init__(self, *, tokenizer: Tokenizer | None = None, planner: VectorizationPlanner | None = None) -> None:
        self.tokenizer = tokenizer or DEFAULT_TOKENIZER
        self.planner = planner or VectorizationPlanner(tokenizer=self.tokenizer)

    def process_paths(
        self,
        paths: Sequence[Path],
        *,
        config: AutoPipelineConfig,
        deps: AutoPipelineDeps | None = None,
    ) -> List[FileProcessingResult]:
        dependencies = deps or AutoPipelineDeps()
        results: List[FileProcessingResult] = []
        for path in paths:
            try:
                result = self._process_single(path, config=config, deps=dependencies)
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception("Failed to process %s", path)
                suffix = path.suffix.lower().lstrip(".") or "txt"
                context = PlanningContext(
                    embedding_model=config.embedding_model,
                    distance=config.distance,
                    collection_name=config.collection_name,
                    file_type=suffix,
                    hard_cap=config.hard_cap,
                    test_queries=config.test_queries,
                )
                empty_plan = config.manual_plan or build_fallback_plan(blocks=[], context=context)
                report = QualityReport(
                    ok=False,
                    issues=[f"Unhandled error: {exc}"],
                    metrics=self._empty_metrics(),
                    recall={},
                )
                results.append(
                    FileProcessingResult(
                        path=path,
                        plan=empty_plan,
                        plan_hash="",
                        quality=report,
                        ingested=False,
                        escalated=True,
                        alias_swapped=False,
                        preview=[],
                        plan_path=None,
                        preview_path=None,
                        metrics_path=None,
                        status="error",
                        notes=[str(exc)],
                    )
                )
            else:
                results.append(result)
        return results

    def _process_single(
        self,
        path: Path,
        *,
        config: AutoPipelineConfig,
        deps: AutoPipelineDeps,
    ) -> FileProcessingResult:
        blocks = normalize_document(path)
        suffix = path.suffix.lower().lstrip(".") or "txt"

        if config.run_mode == "manual":
            if config.manual_plan is None:
                raise ValueError("manual_plan must be provided when run_mode='manual'")
            plan = config.manual_plan
            preview = self.planner.preview_from_plan(
                blocks=blocks,
                plan=plan,
                sample_size=config.preview_limit,
            )
        else:
            context = PlanningContext(
                embedding_model=config.embedding_model,
                distance=config.distance,
                collection_name=config.collection_name,
                file_type=suffix,
                hard_cap=config.hard_cap,
                test_queries=config.test_queries,
            )
            preview = self.planner.analyze(blocks=blocks, context=context)
            plan = preview.plan

        chunk_result = apply_plan(blocks, plan, tokenizer=self.tokenizer)
        preview_data = [
            {
                "text": chunk.text[:300],
                "tokens": chunk.tokens,
                "start": chunk.start,
                "end": chunk.end,
                "meta": chunk.meta,
                "chunk_id": chunk.chunk_id,
            }
            for chunk in chunk_result.chunks[: config.preview_limit]
        ]

        quality = run_quality_checks(
            blocks,
            chunk_result,
            plan,
            coverage_threshold=config.coverage_threshold,
        )

        plan_hash = hashlib.sha256(
            json.dumps(plan.model_dump(), sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

        plan_path, preview_path, metrics_path = self._persist_artifacts(
            path=path,
            plan=plan,
            plan_hash=plan_hash,
            preview_data=preview_data,
            quality=quality,
            config=config,
        )

        notes: List[str] = []
        if plan.collection_name != config.collection_name:
            notes.append(
                "Plan collection_name differs from configured target; using plan value."
            )
        ingested = False
        escalated = False
        alias_swapped = False

        if config.run_mode == "auto":
            if quality.ok:
                ingestion_ok, alias_swapped, ingest_notes = self._ingest(
                    chunk_result=chunk_result,
                    plan=plan,
                    plan_hash=plan_hash,
                    deps=deps,
                )
                notes.extend(ingest_notes)
                ingested = ingestion_ok
                if not ingestion_ok:
                    escalated = True
            else:
                notes.extend(quality.issues)
                escalated = True
        else:
            escalated = True
            if config.run_mode == "review":
                notes.append("Awaiting review approval.")
            else:
                notes.append("Manual mode requires user-supplied ingestion.")

        status = "ingested" if ingested else "escalated" if escalated else "pending"

        return FileProcessingResult(
            path=path,
            plan=plan,
            plan_hash=plan_hash,
            quality=quality,
            ingested=ingested,
            escalated=escalated,
            alias_swapped=alias_swapped,
            preview=preview_data,
            plan_path=plan_path,
            preview_path=preview_path,
            metrics_path=metrics_path,
            status=status,
            notes=notes,
        )

    def _ingest(
        self,
        *,
        chunk_result: ChunkerResult,
        plan: VectorizationPlan,
        plan_hash: str,
        deps: AutoPipelineDeps,
    ) -> tuple[bool, bool, List[str]]:
        notes: List[str] = []
        chunks = [
            AutoChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                tokens=chunk.tokens,
                start=chunk.start,
                end=chunk.end,
                meta=chunk.meta,
            )
            for chunk in chunk_result.chunks
        ]
        texts = [chunk.text for chunk in chunk_result.chunks]
        if not chunks:
            notes.append("No chunks to ingest.")
            return False, False, notes
        if deps.embed_batch is None or deps.upsert_batch is None:
            notes.append("Embedding or upsert dependency not configured; escalation required.")
            return False, False, notes
        try:
            vectors = deps.embed_batch(texts, plan.embedding_model)
        except Exception as exc:  # pragma: no cover - network errors
            notes.append(f"Embedding failed: {exc}")
            return False, False, notes
        if len(vectors) != len(chunks):
            notes.append("Embedding provider returned mismatched vector count.")
            return False, False, notes
        try:
            deps.upsert_batch(chunks, vectors, plan, plan_hash)
        except Exception as exc:  # pragma: no cover - network errors
            notes.append(f"Upsert failed: {exc}")
            return False, False, notes
        alias_swapped = False
        if deps.alias_swap is not None:
            try:
                deps.alias_swap(plan.collection_name)
                alias_swapped = True
            except Exception as exc:  # pragma: no cover - alias errors
                notes.append(f"Alias swap failed: {exc}")
        return True, alias_swapped, notes

    def _persist_artifacts(
        self,
        *,
        path: Path,
        plan: VectorizationPlan,
        plan_hash: str,
        preview_data: List[Dict[str, object]],
        quality: QualityReport,
        config: AutoPipelineConfig,
    ) -> tuple[Path, Path, Path]:
        collection_dir = config.output_dir / plan.collection_name
        collection_dir.mkdir(parents=True, exist_ok=True)
        stem = path.stem or "document"
        plan_path = collection_dir / f"{stem}.plan.json"
        preview_path = collection_dir / f"{stem}.preview.json"
        metrics_path = collection_dir / f"{stem}.metrics.json"

        plan_payload = plan.model_dump()
        plan_payload["plan_hash"] = plan_hash
        plan_path.write_text(
            json.dumps(plan_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        preview_payload = {
            "plan_hash": plan_hash,
            "samples": preview_data,
        }
        preview_path.write_text(
            json.dumps(preview_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        metrics_payload = {
            "plan_hash": plan_hash,
            "plan_source": plan.plan_source,
            "quality": {
                "ok": quality.ok,
                "issues": quality.issues,
                "metrics": asdict(quality.metrics),
                "recall": quality.recall,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        metrics_path.write_text(
            json.dumps(metrics_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return plan_path, preview_path, metrics_path

    @staticmethod
    def _empty_metrics() -> QualityMetrics:  # pragma: no cover - used in error handling only
        return QualityMetrics(
            chunk_count=0,
            avg_tokens=0.0,
            max_tokens=0,
            coverage=0.0,
            token_usage=0,
            duplicate_ranges=0,
        )


__all__ = [
    "AutoChunk",
    "AutoPipelineConfig",
    "AutoPipelineDeps",
    "AutoVectorizationPipeline",
    "FileProcessingResult",
]
