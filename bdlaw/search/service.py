"""Semantic search service."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, List, Optional, Set

from qdrant_client.http import models as rest

from bdlaw.index.embedder import Embedder
from bdlaw.index.vector_store import VectorStore


@dataclass
class SearchResult:
    citation: str
    score: float
    payload: dict


Condition = rest.FieldCondition | rest.IsNullCondition


class SearchService:
    def __init__(self, embedder: Embedder, store: VectorStore):
        self.embedder = embedder
        self.store = store

    def search(
        self,
        query: str,
        k: int = 5,
        law_code: Optional[str] = None,
        on_date: Optional[str] = None,
    ) -> List[SearchResult]:
        vector = self.embedder.embed_batch([query])[0]
        base_filter = self._build_filter(law_code=law_code)
        base_results = self.store.search(vector, k=k, filters=base_filter)
        filtered_results = self._apply_date_filter(base_results, on_date)
        enriched = list(filtered_results)
        for point in filtered_results:
            payload = point.payload or {}
            for cross_ref in payload.get("cross_refs", []):
                extra_filter = self._build_filter(
                    law_code=cross_ref.get("law_code") or payload.get("law_code"),
                    article=cross_ref.get("article"),
                    part=cross_ref.get("part"),
                )
                if extra_filter is None:
                    continue
                extra = self.store.search(vector, k=1, filters=extra_filter)
                for extra_point in self._apply_date_filter(extra, on_date):
                    enriched.append(
                        rest.ScoredPoint(
                            id=extra_point.id,
                            score=extra_point.score * 0.95,
                            payload=extra_point.payload,
                            version=extra_point.version,
                            shard_key=extra_point.shard_key,
                        )
                    )
        return self._deduplicate_results(enriched)

    def _build_filter(
        self,
        *,
        law_code: Optional[str] = None,
        article: Optional[str] = None,
        part: Optional[str] = None,
    ) -> Optional[rest.Filter]:
        must: List[Condition] = []
        if law_code:
            must.append(rest.FieldCondition(key="law_code", match=rest.MatchValue(value=law_code)))
        if article:
            must.append(rest.FieldCondition(key="article", match=rest.MatchValue(value=article)))
        if part:
            must.append(rest.FieldCondition(key="part", match=rest.MatchValue(value=part)))
        if not must:
            return None
        return rest.Filter(must=must)

    def _deduplicate_results(self, points: Iterable[rest.ScoredPoint]) -> List[SearchResult]:
        seen: Set[str] = set()
        results: List[SearchResult] = []
        for point in points:
            payload = point.payload or {}
            key = payload.get("gid") or payload.get("citation")
            if not key or key in seen:
                continue
            seen.add(key)
            results.append(
                SearchResult(
                    citation=payload.get("citation", ""),
                    score=point.score,
                    payload=payload,
                )
            )
        return results

    def _apply_date_filter(
        self, points: Iterable[rest.ScoredPoint], on_date: Optional[str]
    ) -> List[rest.ScoredPoint]:
        if not on_date:
            return list(points)
        target = _parse_date(on_date)
        filtered: List[rest.ScoredPoint] = []
        for point in points:
            payload = point.payload or {}
            valid_from = _parse_date(payload.get("valid_from"))
            valid_to = _parse_date(payload.get("valid_to"))
            if valid_from and valid_from > target:
                continue
            if valid_to and valid_to < target:
                continue
            filtered.append(point)
        return filtered


def _parse_date(value: Optional[str]) -> Optional[date]:
    if value in (None, ""):
        return None
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


__all__ = ["SearchResult", "SearchService"]
