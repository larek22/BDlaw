"""Quality checks for deterministic vectorization plans."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence

from .chunker import Chunk, ChunkerResult
from .models import VectorizationPlan
from .normalizer import Block


@dataclass(slots=True)
class QualityMetrics:
    chunk_count: int
    avg_tokens: float
    max_tokens: int
    coverage: float
    token_usage: int
    duplicate_ranges: int


@dataclass(slots=True)
class QualityReport:
    ok: bool
    issues: List[str]
    metrics: QualityMetrics
    recall: Dict[str, bool]


def run_quality_checks(
    blocks: Sequence[Block],
    chunk_result: ChunkerResult,
    plan: VectorizationPlan,
    *,
    coverage_threshold: float = 0.95,
) -> QualityReport:
    issues: List[str] = []
    total_chars = sum(len((block.text or "").strip()) for block in blocks)
    chunk_chars = sum(len((chunk.text or "").strip()) for chunk in chunk_result.chunks)
    coverage = (chunk_chars / total_chars) if total_chars else 1.0
    chunk_count = len(chunk_result.chunks)
    avg_tokens = (
        sum(chunk.tokens for chunk in chunk_result.chunks) / chunk_count
        if chunk_count
        else 0.0
    )
    max_tokens = max((chunk.tokens for chunk in chunk_result.chunks), default=0)

    if chunk_count < plan.quality_checks.min_chunks:
        issues.append(
            f"Only {chunk_count} chunk(s) generated (minimum {plan.quality_checks.min_chunks})."
        )
    if max_tokens > plan.quality_checks.max_tokens_per_chunk:
        issues.append(
            "Chunk token count exceeds configured maximum"
        )
    if coverage < coverage_threshold:
        issues.append(
            f"Chunk coverage below threshold ({coverage:.2%} < {coverage_threshold:.0%})."
        )

    seen_ranges: set[tuple[int, int]] = set()
    duplicate_ranges = 0
    for chunk in chunk_result.chunks:
        key = (chunk.start, chunk.end)
        if key in seen_ranges:
            duplicate_ranges += 1
        else:
            seen_ranges.add(key)
    if plan.quality_checks.dedupe_near_duplicates and duplicate_ranges:
        issues.append(f"Detected {duplicate_ranges} duplicate chunk ranges.")

    recall = _quick_recall_check(chunk_result.chunks, plan.test_queries)
    if plan.test_queries and not all(recall.values()):
        issues.append("One or more recall probes did not match any chunk text.")

    metrics = QualityMetrics(
        chunk_count=chunk_count,
        avg_tokens=avg_tokens,
        max_tokens=max_tokens,
        coverage=coverage,
        token_usage=chunk_result.token_usage,
        duplicate_ranges=duplicate_ranges,
    )
    return QualityReport(ok=not issues, issues=issues, metrics=metrics, recall=recall)


def _quick_recall_check(chunks: Sequence[Chunk], queries: Sequence[str] | None) -> Dict[str, bool]:
    results: Dict[str, bool] = {}
    if not queries:
        return results
    for query in list(queries)[:5]:
        normalized_terms = [term for term in re.findall(r"\w+", query.lower()) if len(term) > 2]
        if not normalized_terms:
            normalized_terms = [query.lower()]
        found = False
        for chunk in chunks:
            haystack = chunk.text.lower()
            if all(term in haystack for term in normalized_terms):
                found = True
                break
        results[query] = found
    return results


__all__ = ["QualityMetrics", "QualityReport", "run_quality_checks"]
