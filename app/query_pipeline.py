from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Sequence

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

        sources: List[SourceChunk] = []
        chunks: List[Chunk] = []
        for result in search_results:
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
            sources.append(SourceChunk(chunk=chunk, score=result.score))
            chunks.append(chunk)

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
        configured_model = self.settings.openai_models.chat or "gpt-4o-mini"
        try:
            response = self._client.chat.completions.create(
                model=configured_model,
                messages=messages,
                temperature=0,
            )
        except PermissionDeniedError as exc:
            fallback_model = "gpt-4o-mini"
            if configured_model != fallback_model:
                logger.warning(
                    "Permission denied for OpenAI model %s; retrying with fallback %s",
                    configured_model,
                    fallback_model,
                )
                try:
                    response = self._client.chat.completions.create(
                        model=fallback_model,
                        messages=messages,
                        temperature=0,
                    )
                except PermissionDeniedError as fallback_exc:
                    raise RuntimeError(
                        "OpenAI denied access to both the configured chat model and fallback gpt-4o-mini. "
                        "Please choose a model available to your account in the Settings tab."
                    ) from fallback_exc
                except OpenAIError as fallback_exc:
                    raise RuntimeError(
                        "OpenAI chat completion failed when using fallback model gpt-4o-mini: "
                        f"{fallback_exc}"
                    ) from fallback_exc
                else:
                    self.settings.openai_models.chat = fallback_model
                    try:
                        self.settings.save()
                    except Exception:
                        logger.exception("Failed to persist fallback chat model selection")
            else:
                raise RuntimeError(
                    "OpenAI denied access to the configured chat model. "
                    "Please choose a model available to your account in the Settings tab."
                ) from exc
        except OpenAIError as exc:
            raise RuntimeError(f"OpenAI chat completion failed: {exc}") from exc

        answer = response.choices[0].message.content or "I don't know."
        return QueryResponse(answer=answer.strip(), sources=sources)
