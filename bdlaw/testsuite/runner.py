"""Quality evaluation engine."""

from __future__ import annotations

import json
from dataclasses import dataclass
from math import log2
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Sequence

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
    ndcg_at_k: float
    answer_pass_at_k: float


class TestSuiteRunner:
    def __init__(
        self,
        search: SearchService,
        k: int = 5,
        answer_validator: Optional[object] = None,
    ):
        self.search = search
        self.k = k
        self.answer_validator = answer_validator

    def run(self, suite: Sequence[GoldenQuery]) -> MetricReport:
        recalls: List[float] = []
        precisions: List[float] = []
        reciprocal_ranks: List[float] = []
        ndcgs: List[float] = []
        passes: List[float] = []
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
            ndcgs.append(self._ndcg(golden, results))
            passes.append(self._answer_pass(golden, results))
        return MetricReport(
            recall_at_k=mean(recalls) if recalls else 0.0,
            precision_at_k=mean(precisions) if precisions else 0.0,
            mrr_at_k=mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
            ndcg_at_k=mean(ndcgs) if ndcgs else 0.0,
            answer_pass_at_k=mean(passes) if passes else 0.0,
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

    def _ndcg(self, golden: GoldenQuery, results: Sequence[SearchResult]) -> float:
        if not golden.expected_norms:
            return 1.0
        relevance = [self._relevance(result, golden.expected_norms) for result in results]
        dcg = sum((rel / log2(idx + 2)) for idx, rel in enumerate(relevance))
        ideal = sorted(relevance, reverse=True)
        idcg = sum((rel / log2(idx + 2)) for idx, rel in enumerate(ideal))
        return dcg / idcg if idcg else 0.0

    def _answer_pass(self, golden: GoldenQuery, results: Sequence[SearchResult]) -> float:
        if not self.answer_validator or not results:
            return 0.0
        answer = self._build_answer(results)
        citations = [res.payload.get("citation", "") for res in results if res.payload.get("citation")]
        validation = self.answer_validator.validate(
            answer,
            citations,
            [f"{norm.get('law_code', '')} ст.{norm.get('article')}" for norm in golden.expected_norms],
        )
        return 1.0 if validation.verdict.upper() == "PASS" else 0.0

    def _relevance(self, result: SearchResult, expected: Iterable[Dict[str, str]]) -> float:
        for norm in expected:
            if (
                result.payload.get("law_code") == norm.get("law_code")
                and result.payload.get("article") == norm.get("article")
                and result.payload.get("part") == norm.get("part")
            ):
                return 1.0
        return 0.0

    @staticmethod
    def _build_answer(results: Sequence[SearchResult]) -> str:
        lines = []
        for result in results:
            citation = result.payload.get("citation")
            url = result.payload.get("source_url", "")
            if citation and url:
                lines.append(f"[{citation}] — Автоматическая проверка. Источник: {url}")
        return "\n".join(lines) if lines else "Недостаточно данных"

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
