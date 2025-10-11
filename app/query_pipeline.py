from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from openai import OpenAI, OpenAIError, PermissionDeniedError
from qdrant_client.http import models as rest

from .embeddings import EmbeddingClient
from .legal_types import ChunkRecord
from .qdrant_client import QdrantVectorStore
from .settings import AppSettings

logger = logging.getLogger(__name__)


@dataclass
class SourceChunk:
    chunk: ChunkRecord
    score: float


@dataclass
class QueryResponse:
    answer: str
    sources: List[SourceChunk]


class QueryPipeline:
    def __init__(self, settings: AppSettings, embedding_client: EmbeddingClient, vector_store: QdrantVectorStore) -> None:
        self.settings = settings
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        api_key = settings.openai_api_key or None
        self._client = OpenAI(api_key=api_key)
        self._chat_model: str | None = None

    def _format_context(self, sources: Sequence[SourceChunk]) -> str:
        lines = []
        for idx, source in enumerate(sources, start=1):
            chunk = source.chunk
            hierarchy = chunk.hierarchy or {}
            trail = []
            if hierarchy.get("part_no"):
                trail.append(f"Part {hierarchy['part_no']}")
            if hierarchy.get("article_no"):
                trail.append(f"Article {hierarchy['article_no']}")
            header = chunk.title_text or ", ".join(trail)
            lines.append(
                f"[{idx}] {header}\n{chunk.body_text}"
            )
        return "\n\n".join(lines)

    def answer(self, question: str, top_k: int | None = None) -> QueryResponse:
        top_k = top_k or self.settings.query.top_k
        sources = self._retrieve_sources(question, top_k)
        context = self._format_context(sources)
        prompt = (
            "You are a helpful assistant. Answer strictly using the provided context. "
            "If the answer is not in the context, say you don't know."
        )
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ]
        model = self._get_chat_model()
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
            )
        except PermissionDeniedError as exc:
            self._chat_model = None
            raise RuntimeError(
                "OpenAI denied access to the selected chat model. "
                "Please choose a model available to your account in the Settings tab."
            ) from exc
        except OpenAIError as exc:
            raise RuntimeError(f"OpenAI chat completion failed: {exc}") from exc

        answer = response.choices[0].message.content or "I don't know."
        return QueryResponse(answer=answer.strip(), sources=sources)

    def test_retrieval(self, question: str = "retrieval test", top_k: int = 3) -> List[SourceChunk]:
        """Perform a lightweight retrieval to validate search results."""

        limit = max(1, top_k or self.settings.query.top_k)
        return self._retrieve_sources(question, limit)

    def retrieve(self, question: str, top_k: int | None = None) -> List[SourceChunk]:
        top_k = top_k or self.settings.query.top_k
        return self._retrieve_sources(question, top_k)

    def _retrieve_sources(self, question: str, top_k: int) -> List[SourceChunk]:
        query_vector = self.embedding_client.embed_query(question)
        filter_ = self._build_combined_filter(question, top_k)
        body_results = self.vector_store.query(
            vector_name="body_vec",
            query_vector=query_vector,
            limit=max(top_k * 5, top_k + 40),
            filters=filter_,
        )
        title_results = self.vector_store.query(
            vector_name="title_vec",
            query_vector=query_vector,
            limit=max(top_k * 3, top_k + 20),
            filters=filter_,
        )
        fused_results = self._fuse_results(
            body_results,
            title_results,
            body_weight=self.settings.query.fusion_weight_body,
            title_weight=self.settings.query.fusion_weight_title,
            rrf_k=max(1, self.settings.query.fusion_rrf_k),
        )
        return self._hydrate_sources(fused_results[:top_k])

    def _build_combined_filter(self, question: str, top_k: int) -> rest.Filter | None:
        filters: List[rest.Filter] = []
        prefilter_limit = max(top_k * 5, self.settings.query.prefilter_limit)
        if prefilter_limit <= 0:
            prefilter = None
        else:
            doc_ids = self.vector_store.keyword_prefilter(question, prefilter_limit)
            prefilter = None
            if doc_ids:
                try:
                    prefilter = self.vector_store.build_doc_id_filter(doc_ids)
                except ValueError:
                    prefilter = None
        if prefilter:
            filters.append(prefilter)

        temporal_filter = self.vector_store.build_as_of_filter(
            status=self.settings.query.status_filter,
            as_of_start=self.settings.query.as_of_start_date,
            as_of_date=self.settings.query.as_of_date,
        )
        if temporal_filter:
            filters.append(temporal_filter)

        return self.vector_store.combine_filters(*filters)

    def _fuse_results(
        self,
        body_results: Sequence,
        title_results: Sequence,
        *,
        body_weight: float,
        title_weight: float,
        rrf_k: int,
    ) -> List:
        combined: dict[str, float] = {}
        registry: dict[str, object] = {}
        for weight, results in ((body_weight, body_results), (title_weight, title_results)):
            if weight <= 0:
                continue
            for rank, result in enumerate(results, start=1):
                point_id = str(getattr(result, "id", ""))
                if not point_id:
                    continue
                registry[point_id] = result
                combined[point_id] = combined.get(point_id, 0.0) + weight * (1.0 / (rrf_k + rank))
        sorted_ids = sorted(combined.items(), key=lambda item: item[1], reverse=True)
        return [registry[point_id] for point_id, _ in sorted_ids]

    def _hydrate_sources(self, results: Iterable) -> List[SourceChunk]:
        sources: List[SourceChunk] = []
        for result in results:
            payload = result.payload or {}

            raw_title = payload.get("title_text")
            title_text = raw_title if isinstance(raw_title, str) else ""
            raw_body = payload.get("body_text")
            body_text = raw_body if isinstance(raw_body, str) else ""

            if not title_text and body_text:
                title_text = body_text[:120].strip()
            if not body_text and title_text:
                body_text = title_text

            chunk = ChunkRecord(
                doc_id=str(payload.get("doc_id") or ""),
                chunk_index=int(payload.get("chunk_index") or 0),
                chunk_id=str(payload.get("chunk_id") or ""),
                title_text=title_text,
                body_text=body_text,
                hierarchy=payload.get("hierarchy") or {},
                law_meta=payload.get("law_meta") or {},
                source=payload.get("source") or {},
                chunk_key=payload.get("chunk_key"),
                plan_version=payload.get("plan_version"),
                parser_version=payload.get("parser_version"),
                chunk_sha256=str(payload.get("chunk_sha256") or ""),
                title_sha256=str(payload.get("title_sha256") or ""),
                body_sha256=str(payload.get("body_sha256") or ""),
            )
            sources.append(SourceChunk(chunk=chunk, score=float(getattr(result, "score", 0.0) or 0.0)))
        return sources

    def _get_chat_model(self) -> str:
        if self._chat_model:
            return self._chat_model
        self._chat_model = self._resolve_chat_model()
        return self._chat_model

    def _resolve_chat_model(self) -> str:
        preferred = (self.settings.openai_models.chat or "").strip()
        candidates = []
        for candidate in [preferred, "gpt-4.1-mini", "gpt-4o-mini"]:
            if candidate and candidate not in candidates:
                candidates.append(candidate)

        last_error: Exception | None = None
        probe_messages = [{"role": "user", "content": "ping"}]

        for model in candidates:
            try:
                self._client.chat.completions.create(
                    model=model,
                    messages=probe_messages,
                    max_tokens=1,
                    temperature=0,
                )
            except PermissionDeniedError as exc:
                logger.warning(
                    "Permission denied for OpenAI model %s during probe: %s", model, exc
                )
                last_error = exc
                continue
            except OpenAIError as exc:
                logger.error(
                    "OpenAI chat probe failed for model %s: %s", model, exc
                )
                raise RuntimeError(
                    f"OpenAI chat completion failed during model probe: {exc}"
                ) from exc
            else:
                logger.info("Using OpenAI chat model: %s", model)
                if model != preferred:
                    self.settings.openai_models.chat = model
                    try:
                        self.settings.save()
                    except Exception:
                        logger.exception("Failed to persist resolved chat model")
                return model

        raise RuntimeError(
            "No accessible OpenAI chat model found. Please update the Chat Model in Settings."
        ) from last_error
