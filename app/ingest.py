from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterable, List, Sequence

from .chunker import Chunk, chunk_document
from .settings import AppSettings
from .qdrant_client import QdrantVectorStore

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .embeddings import EmbeddingClient
    from .readers.factory import DocumentReaderFactory

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
        embedding_client: "EmbeddingClient",
        vector_store: QdrantVectorStore,
        reader_factory: "DocumentReaderFactory" | None = None,
    ) -> None:
        self.settings = settings
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        if reader_factory is None:
            from .readers.factory import DocumentReaderFactory  # local import to avoid optional deps at import time

            reader_factory = DocumentReaderFactory()
        self.reader_factory = reader_factory

    def ingest(
        self,
        paths: Sequence[Path],
        recreate: bool = False,
        *,
        progress_cb: Callable[[str], None] | None = None,
    ) -> IngestStats:
        def report(message: str, *, level: int = logging.INFO) -> None:
            logger.log(level, message)
            if progress_cb:
                progress_cb(message)

        chunk_size = self.settings.ingest.chunk_size_chars
        overlap = self.settings.ingest.chunk_overlap_chars

        report(
            f"Preparing ingestion for {len(paths)} file(s) (chunk_size={chunk_size} overlap={overlap})",
            level=logging.INFO,
        )

        documents: List[tuple[Path, List[Chunk]]] = []
        skipped = 0

        for path in paths:
            try:
                document = self.reader_factory.read(path)
            except Exception as exc:
                message = f"Failed to read {path}: {exc}"
                logger.error(message)
                if progress_cb:
                    progress_cb(message)
                skipped += 1
                continue
            chunks = chunk_document(document, chunk_size, overlap)
            if not chunks:
                report(
                    f"No chunks produced for {path.name}; skipping",
                    level=logging.WARNING,
                )
                skipped += 1
                continue
            report(
                f"Parsed {path.name} → {len(chunks)} chunk(s)",
                level=logging.INFO,
            )
            documents.append((path, chunks))

        all_chunks: List[Chunk] = []
        for _, chunks in documents:
            all_chunks.extend(chunks)

        if not all_chunks:
            return IngestStats(files_processed=0, chunks_created=0, skipped=skipped)

        embedding_model = self.settings.openai_models.embedding
        report(
            f"Ensuring collection '{self.vector_store.collection_name}' for embedding model {embedding_model}",
            level=logging.INFO,
        )
        self.vector_store.ensure_collection(embedding_model=embedding_model, recreate=recreate)

        report(
            f"Embedding {len(all_chunks)} chunk(s) produced from {len(documents)} file(s)",
            level=logging.INFO,
        )

        sample_chunk = all_chunks[0]
        preview = sample_chunk.text.replace("\n", " ").strip()
        if len(preview) > 120:
            preview = preview[:117] + "…"
        report(
            f"Sample chunk: doc={sample_chunk.doc_id} index={sample_chunk.chunk_index} sha={sample_chunk.sha[:12]} preview='{preview}'",
            level=logging.INFO,
        )

        texts = [chunk.text for chunk in all_chunks]
        shas = [chunk.sha for chunk in all_chunks]
        embeddings = self.embedding_client.embed_texts(texts, shas)

        sha_to_vector = {result.sha: result.vector for result in embeddings}
        vectors = [sha_to_vector[chunk.sha] for chunk in all_chunks]
        vector_dim = len(vectors[0]) if vectors else 0
        expected_dim = self.vector_store.vector_size_for_model(embedding_model)
        if vectors and vector_dim != expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {expected_dim} from {embedding_model} "
                f"but received {vector_dim}"
            )
        report(
            f"[UPSERT] collection={self.vector_store.collection_name} points={len(vectors)} dim={vector_dim}",
            level=logging.INFO,
        )

        before_count: int | None = None
        try:
            before_count = self.vector_store.count_points()
        except Exception as exc:
            message = f"Could not read existing point count: {exc}"
            logger.warning(message)
            if progress_cb:
                progress_cb(message)
        else:
            report(
                f"[QDRANT] existing points before ingest: {before_count}",
                level=logging.INFO,
            )

        try:
            self.vector_store.upsert_chunks(
                all_chunks,
                vectors,
                embedding_model=embedding_model,
                chunk_size=chunk_size,
                chunk_overlap=overlap,
            )
        except Exception:
            logger.exception("Failed to upsert vectors to Qdrant")
            raise

        try:
            total_points = self.vector_store.count_points()
        except Exception:
            logger.warning("Unable to retrieve Qdrant point count after upsert")
        else:
            report(f"[QDRANT] total points now: {total_points}", level=logging.INFO)
            if before_count is not None:
                delta = total_points - before_count
                report(
                    f"[QDRANT] points added or updated in this run: {delta}",
                    level=logging.INFO,
                )

        if vectors:
            try:
                verify_hits = self.vector_store.search(vectors[0], top_k=1)
            except Exception as exc:
                message = f"[VERIFY] sample search failed: {exc}"
                logger.warning(message)
                if progress_cb:
                    progress_cb(message)
            else:
                if verify_hits:
                    payload = getattr(verify_hits[0], "payload", {}) or {}
                    doc_id = payload.get("doc_id", "<unknown>")
                    chunk_index = payload.get("chunk_index", "?")
                    raw_score = getattr(verify_hits[0], "score", 0.0)
                    score = float(raw_score) if raw_score is not None else 0.0
                    report(
                        f"[VERIFY] top match doc={doc_id} chunk={chunk_index} score={score:.4f}",
                        level=logging.INFO,
                    )
                else:
                    report("[VERIFY] sample search returned no hits", level=logging.WARNING)

        return IngestStats(
            files_processed=len(documents),
            chunks_created=len(all_chunks),
            skipped=skipped,
        )
