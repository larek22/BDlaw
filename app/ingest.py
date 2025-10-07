from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from .chunker import Chunk, chunk_document
from .embeddings import EmbeddingClient
from .readers.factory import DocumentReaderFactory
from .settings import AppSettings
from .qdrant_client import QdrantVectorStore

logger = logging.getLogger(__name__)


@dataclass
class IngestStats:
    files_processed: int
    chunks_created: int
    skipped: int


class IngestService:
    def __init__(
        self,
        settings: AppSettings,
        embedding_client: EmbeddingClient,
        vector_store: QdrantVectorStore,
        reader_factory: DocumentReaderFactory | None = None,
    ) -> None:
        self.settings = settings
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        self.reader_factory = reader_factory or DocumentReaderFactory()

    def ingest(self, paths: Sequence[Path], recreate: bool = False) -> IngestStats:
        chunk_size = self.settings.ingest.chunk_size_chars
        overlap = self.settings.ingest.chunk_overlap_chars

        documents: List[tuple[Path, List[Chunk]]] = []
        skipped = 0

        for path in paths:
            try:
                document = self.reader_factory.read(path)
            except Exception as exc:
                logger.error("Failed to read %s: %s", path, exc)
                skipped += 1
                continue
            chunks = chunk_document(document, chunk_size, overlap)
            if not chunks:
                skipped += 1
                continue
            documents.append((path, chunks))

        all_chunks: List[Chunk] = []
        for _, chunks in documents:
            all_chunks.extend(chunks)

        if not all_chunks:
            return IngestStats(files_processed=0, chunks_created=0, skipped=skipped)

        self.vector_store.ensure_collection(vector_size=3072, recreate=recreate)

        texts = [chunk.text for chunk in all_chunks]
        shas = [chunk.sha for chunk in all_chunks]
        embeddings = self.embedding_client.embed_texts(texts, shas)

        sha_to_vector = {result.sha: result.vector for result in embeddings}
        vectors = [sha_to_vector[chunk.sha] for chunk in all_chunks]
        self.vector_store.upsert_chunks(all_chunks, vectors)

        return IngestStats(
            files_processed=len(documents),
            chunks_created=len(all_chunks),
            skipped=skipped,
        )
