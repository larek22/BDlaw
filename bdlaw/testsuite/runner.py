"""Quality evaluation engine."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Sequence

from bdlaw.search.service import SearchService, SearchResult


@dataclass
class GoldenQuery:
    query: str
    expected_norms: List[Dict[str, str]]
    on_date: str | None = None


@dataclass
class MetricReport:
    recall_at_k: float
    precision_at_k: float
    mrr_at_k: float


class TestSuiteRunner:
    def __init__(self, search: SearchService, k: int = 5):
        self.search = search
        self.k = k

    def run(self, suite: Sequence[GoldenQuery]) -> MetricReport:
        recalls: List[float] = []
        precisions: List[float] = []
        reciprocal_ranks: List[float] = []
        for golden in suite:
            results = self.search.search(
                query=golden.query,
                k=self.k,
                law_code=golden.expected_norms[0].get("law_code") if golden.expected_norms else None,
                on_date=golden.on_date,
            )
            recalls.append(self._recall(golden, results))
            precisions.append(self._precision(golden, results))
            reciprocal_ranks.append(self._mrr(golden, results))
        return MetricReport(
            recall_at_k=mean(recalls) if recalls else 0.0,
            precision_at_k=mean(precisions) if precisions else 0.0,
            mrr_at_k=mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        )

    @staticmethod
    def load_suite(path: Path) -> List[GoldenQuery]:
        data = json.loads(path.read_text(encoding="utf-8"))
        suite = [
            GoldenQuery(
                query=item["query"],
                expected_norms=item.get("expected_norms", []),
                on_date=item.get("on_date"),
            )
            for item in data
        ]
        return suite

    def _recall(self, golden: GoldenQuery, results: Sequence[SearchResult]) -> float:
        if not golden.expected_norms:
            return 1.0
        retrieved = self._match_norms(golden, results)
        return len(retrieved) / len(golden.expected_norms)

    def _precision(self, golden: GoldenQuery, results: Sequence[SearchResult]) -> float:
        if not results:
            return 0.0
        retrieved = self._match_norms(golden, results)
        return len(retrieved) / len(results)

    def _mrr(self, golden: GoldenQuery, results: Sequence[SearchResult]) -> float:
        expected = {(norm.get("article"), norm.get("part")) for norm in golden.expected_norms}
        for idx, result in enumerate(results, start=1):
            article = result.payload.get("article")
            part = result.payload.get("part")
            if (article, part) in expected:
                return 1.0 / idx
        return 0.0

    @staticmethod
    def _match_norms(golden: GoldenQuery, results: Sequence[SearchResult]) -> List[SearchResult]:
        expected = {(norm.get("article"), norm.get("part")) for norm in golden.expected_norms}
        matches = [
            result
            for result in results
            if (result.payload.get("article"), result.payload.get("part")) in expected
        ]
        return matches


__all__ = ["GoldenQuery", "MetricReport", "TestSuiteRunner"]
