"""Semantic search service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from qdrant_client.http import models as rest

from bdlaw.index.embedder import Embedder
from bdlaw.index.vector_store import VectorStore


@dataclass
class SearchResult:
    citation: str
    score: float
    payload: dict


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
        filters: Optional[rest.Filter] = None
        must: List[rest.FieldCondition] = []
        if law_code:
            must.append(rest.FieldCondition(key="law_code", match=rest.MatchValue(value=law_code)))
        if on_date:
            must.append(
                rest.FieldCondition(
                    key="valid_from",
                    range=rest.Range(lte=on_date),
                )
            )
            should = [
                rest.FieldCondition(
                    key="valid_to",
                    range=rest.Range(gte=on_date),
                ),
                rest.IsNullCondition(key="valid_to"),
            ]
            filters = rest.Filter(must=must, should=should)
        if filters is None and must:
            filters = rest.Filter(must=must)
        results = self.store.search(vector, k=k, filters=filters)
        return [
            SearchResult(
                citation=point.payload.get("citation", ""),
                score=point.score,
                payload=point.payload,
            )
            for point in results
        ]


__all__ = ["SearchResult", "SearchService"]
