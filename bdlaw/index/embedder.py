"""Embedding interfaces and OpenAI implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List, Sequence

from openai import OpenAI

from bdlaw.settings.config import EmbeddingConfig


class Embedder(ABC):
    @abstractmethod
    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:  # pragma: no cover
        ...


class OpenAIEmbedder(Embedder):
    def __init__(self, config: EmbeddingConfig):
        self.model = config.model
        self.batch_size = config.batch_size
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            try:
                self._client = OpenAI()
            except Exception as exc:  # noqa: BLE001 - surface friendly guidance
                raise RuntimeError(
                    "Не удалось инициализировать OpenAI-клиент. "
                    "Проверьте, что установлен пакет 'openai' и настроен API-ключ (" 
                    "например, переменная окружения OPENAI_API_KEY или запись в keyring)."
                ) from exc
        return self._client

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        client = self._get_client()
        vectors: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            response = client.embeddings.create(model=self.model, input=list(chunk))
            vectors.extend([item.embedding for item in response.data])
        return vectors


__all__ = ["Embedder", "OpenAIEmbedder"]
