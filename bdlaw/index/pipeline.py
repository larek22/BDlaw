"""Indexing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from bdlaw.index.embedder import Embedder
from bdlaw.index.vector_store import VectorStore
from bdlaw.parser.structure import Norm
from bdlaw.settings.config import ChunkingConfig


@dataclass
class IndexingStats:
    norms_indexed: int


class IndexingPipeline:
    def __init__(self, embedder: Embedder, store: VectorStore, chunking: ChunkingConfig):
        self.embedder = embedder
        self.store = store
        self.chunking = chunking

    def index(self, norms: Iterable[Norm]) -> IndexingStats:
        norm_list = list(norms)
        texts: List[str] = [norm.clean_text or norm.raw_text for norm in norm_list]
        vectors = self.embedder.embed_batch(texts)
        self.store.upsert(norm_list, vectors)
        return IndexingStats(norms_indexed=len(norm_list))


__all__ = ["IndexingPipeline", "IndexingStats"]
