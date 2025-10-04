"""Indexing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from bdlaw.index.embedder import Embedder
from bdlaw.index.payload import NormPayload
from bdlaw.index.vector_store import VectorStore
from bdlaw.settings.config import ChunkingConfig


@dataclass
class IndexingStats:
    norms_indexed: int


class IndexingPipeline:
    def __init__(self, embedder: Embedder, store: VectorStore, chunking: ChunkingConfig):
        self.embedder = embedder
        self.store = store
        self.chunking = chunking

    def index(self, norms: Iterable[NormPayload]) -> IndexingStats:
        unique: List[NormPayload] = []
        seen_hashes: set[str] = set()
        for norm in norms:
            if norm.hash in seen_hashes:
                continue
            seen_hashes.add(norm.hash)
            unique.append(norm)
        texts: List[str] = [norm.clean_text or norm.raw_text for norm in unique]
        vectors = self.embedder.embed_batch(texts)
        self.store.upsert(unique, vectors)
        return IndexingStats(norms_indexed=len(unique))


__all__ = ["IndexingPipeline", "IndexingStats"]
