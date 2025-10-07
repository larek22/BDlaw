from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from openai import OpenAI, OpenAIError, PermissionDeniedError

from .chunker import Chunk
from .embeddings import EmbeddingClient
from .qdrant_client import QdrantVectorStore
from .settings import AppSettings

logger = logging.getLogger(__name__)


@dataclass
class SourceChunk:
    chunk: Chunk
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
            location = f"p.{chunk.page}" if chunk.page else f"chunk {chunk.chunk_index}"
            lines.append(
                f"[{idx}] {chunk.doc_id} ({location}): {chunk.text}"
            )
        return "\n".join(lines)

    def answer(self, question: str, top_k: int | None = None) -> QueryResponse:
        top_k = top_k or self.settings.query.top_k
        query_vector = self.embedding_client.embed_query(question)
        search_results = self.vector_store.search(query_vector, top_k)

        sources, _ = self._hydrate_sources(search_results)
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
        query_vector = self.embedding_client.embed_query(question)
        search_results = self.vector_store.search(query_vector, limit)
        sources, _ = self._hydrate_sources(search_results)
        return sources

    def _hydrate_sources(self, results: Iterable) -> tuple[List[SourceChunk], List[Chunk]]:
        sources: List[SourceChunk] = []
        chunks: List[Chunk] = []
        for result in results:
            payload = result.payload or {}
            chunk = Chunk(
                doc_id=payload.get("doc_id", ""),
                path=payload.get("path", ""),
                page=payload.get("page"),
                chunk_index=payload.get("chunk_index", 0),
                offset=payload.get("offset", 0),
                text=payload.get("text", ""),
                sha=payload.get("sha", ""),
            )
            sources.append(SourceChunk(chunk=chunk, score=getattr(result, "score", 0.0)))
            chunks.append(chunk)
        return sources, chunks

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
