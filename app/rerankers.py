from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Iterable, List, Sequence

try:  # pragma: no cover - optional dependency
    from sentence_transformers import CrossEncoder  # type: ignore
except Exception:  # pragma: no cover - runtime without optional dep
    CrossEncoder = None  # type: ignore

from .settings import AppSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RerankCandidate:
    candidate_id: str
    title: str
    body: str
    base_score: float

    def text(self) -> str:
        if self.title and self.body:
            return f"{self.title}\n\n{self.body}"
        return self.body or self.title


class BaseReranker:
    def rerank(self, query: str, candidates: Sequence[RerankCandidate]) -> List[float]:
        raise NotImplementedError


class PassthroughReranker(BaseReranker):
    def rerank(self, query: str, candidates: Sequence[RerankCandidate]) -> List[float]:
        return [candidate.base_score for candidate in candidates]


class KeywordReranker(BaseReranker):
    def __init__(self, *, weight: float) -> None:
        self.weight = max(0.0, min(1.0, weight))

    def rerank(self, query: str, candidates: Sequence[RerankCandidate]) -> List[float]:
        terms = [term for term in query.lower().split() if term]
        if not candidates or not terms:
            return [candidate.base_score for candidate in candidates]
        scores: List[float] = []
        for candidate in candidates:
            text = candidate.text().lower()
            hits = 0.0
            if text:
                hits = sum(1.0 for term in terms if term in text) / len(terms)
            fused = (1.0 - self.weight) * candidate.base_score + self.weight * hits
            scores.append(fused)
        return scores


class CachedReranker(BaseReranker):
    def __init__(self, inner: BaseReranker, ttl_seconds: int = 300) -> None:
        self._inner = inner
        self._ttl_seconds = max(0, ttl_seconds)
        self._cache: dict[tuple[str, tuple[str, ...]], tuple[float, List[float]]] = {}

    def rerank(self, query: str, candidates: Sequence[RerankCandidate]) -> List[float]:
        if not self._ttl_seconds:
            return self._inner.rerank(query, candidates)
        key = (query, tuple(candidate.candidate_id for candidate in candidates))
        now = time.monotonic()
        cached = self._cache.get(key)
        if cached and cached[0] > now:
            return list(cached[1])
        scores = self._inner.rerank(query, candidates)
        self._cache[key] = (now + self._ttl_seconds, list(scores))
        return scores


class BgeReranker(BaseReranker):
    def __init__(self, model_name: str, *, batch_size: int = 8) -> None:
        if CrossEncoder is None:  # pragma: no cover - optional dependency missing
            raise RuntimeError("sentence-transformers not installed; BGE reranker unavailable")
        self._model = CrossEncoder(model_name)
        self._batch_size = max(1, batch_size)

    def rerank(self, query: str, candidates: Sequence[RerankCandidate]) -> List[float]:
        if not candidates:
            return []
        pairs = [[query, candidate.text()] for candidate in candidates]
        scores = self._model.predict(pairs, batch_size=self._batch_size)
        return [float(score) for score in scores]


class LlmReranker(BaseReranker):
    def __init__(self, client, model: str) -> None:
        self._client = client
        self._model = model

    def rerank(self, query: str, candidates: Sequence[RerankCandidate]) -> List[float]:  # pragma: no cover - network
        if not candidates:
            return []
        snippets = [candidate.text()[:700] for candidate in candidates]
        prompt_parts = [
            "You are ranking legal text snippets for relevance.",
            "Return a JSON list of relevance scores between 0 and 1 for each snippet in order.",
            "Query: " + query,
        ]
        for idx, snippet in enumerate(snippets, start=1):
            prompt_parts.append(f"Snippet {idx}: {snippet}")
        prompt = "\n\n".join(prompt_parts)
        response = self._client.responses.create(  # type: ignore[attr-defined]
            model=self._model,
            input=prompt,
        )
        choice = response.output[0]
        text = getattr(choice, "text", None)
        if not text:
            return [candidate.base_score for candidate in candidates]
        try:
            payload = json.loads(text)
            if isinstance(payload, list):
                scores = [float(value) for value in payload]
                if len(scores) == len(candidates):
                    return scores
        except Exception:
            logger.debug("LLM reranker returned non-JSON payload")
        return [candidate.base_score for candidate in candidates]


def build_reranker(settings: AppSettings, openai_client) -> BaseReranker:
    query_settings = settings.query
    mode = (query_settings.reranker_mode or "off").lower()
    if mode in {"off", "none"}:
        return PassthroughReranker()
    if mode == "bge":
        try:
            reranker: BaseReranker = BgeReranker(
                query_settings.reranker_model,
                batch_size=query_settings.reranker_batch_size,
            )
        except RuntimeError as exc:
            logger.warning("BGE reranker unavailable: %s; falling back to keyword reranker", exc)
            reranker = KeywordReranker(weight=query_settings.reranker_weight)
    elif mode == "llm":
        if openai_client is None:
            logger.warning("LLM reranker requested but OpenAI client unavailable; using keyword reranker")
            reranker = KeywordReranker(weight=query_settings.reranker_weight)
        else:
            reranker = LlmReranker(openai_client, settings.openai_models.chat)
    else:
        reranker = KeywordReranker(weight=query_settings.reranker_weight)

    if query_settings.reranker_cache_ttl_seconds > 0:
        reranker = CachedReranker(reranker, ttl_seconds=query_settings.reranker_cache_ttl_seconds)
    return reranker


__all__ = [
    "BaseReranker",
    "BgeReranker",
    "CachedReranker",
    "KeywordReranker",
    "LlmReranker",
    "PassthroughReranker",
    "RerankCandidate",
    "build_reranker",
]
