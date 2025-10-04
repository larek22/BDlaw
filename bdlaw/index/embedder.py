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
        self.client = OpenAI()
        self.model = config.model
        self.batch_size = config.batch_size

    def embed_batch(self, texts: Sequence[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start : start + self.batch_size]
            response = self.client.embeddings.create(model=self.model, input=list(chunk))
            vectors.extend([item.embedding for item in response.data])
        return vectors


__all__ = ["Embedder", "OpenAIEmbedder"]
