from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Sequence

from .data_repository import DataRepository
from .legal_pipeline import LegalCorpusBuilder, ProcessedDocument, normalize_text
from .legal_types import ChunkRecord
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
        repository: DataRepository | None = None,
    ) -> None:
        self.settings = settings
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        self.repository = repository or DataRepository()
        if reader_factory is None:
            from .readers.factory import DocumentReaderFactory  # local import to avoid optional deps at import time

            reader_factory = DocumentReaderFactory()
        self.reader_factory = reader_factory
        self.builder = LegalCorpusBuilder(self.repository)

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

        skipped = 0
        processed_documents: List[ProcessedDocument] = []

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
            normalized = normalize_text(document.full_text)
            processed = self.builder.process_document(
                document,
                normalized_text=normalized,
                chunk_size=chunk_size,
                overlap=overlap,
            )
            report(
                f"Parsed {path.name}: {len(processed.articles)} article(s), {len(processed.chunks)} chunk(s)",
                level=logging.INFO,
            )
            if processed.doc_ids_changed:
                report(
                    "Changed doc ids: " + ", ".join(processed.doc_ids_changed),
                    level=logging.INFO,
                )
            if processed.doc_ids_unchanged:
                report(
                    "Unchanged doc ids: " + ", ".join(processed.doc_ids_unchanged),
                    level=logging.INFO,
                )
            processed_documents.append(processed)

        changed_chunks: List[ChunkRecord] = []
        changed_doc_ids: set[str] = set()
        unchanged_doc_ids: set[str] = set()
        for processed in processed_documents:
            changed_doc_ids.update(processed.doc_ids_changed)
            unchanged_doc_ids.update(processed.doc_ids_unchanged)
            changed_set = set(processed.doc_ids_changed)
            for chunk in processed.chunks:
                if chunk.doc_id in changed_set:
                    changed_chunks.append(chunk)

        if not changed_chunks:
            return IngestStats(files_processed=0, chunks_created=0, skipped=skipped)

        embedding_model = self.settings.openai_models.embedding
        report(
            f"Ensuring collection '{self.vector_store.collection_name}' for embedding model {embedding_model}",
            level=logging.INFO,
        )
        self.vector_store.ensure_collection(embedding_model=embedding_model, recreate=recreate)

        if changed_doc_ids:
            report(
                f"Deleting {len(changed_doc_ids)} document id(s) prior to upsert",
                level=logging.INFO,
            )
            self.vector_store.delete_documents(sorted(changed_doc_ids))

        report(
            f"Embedding {len(changed_chunks)} chunk(s) from {len(processed_documents)} file(s)",
            level=logging.INFO,
        )

        sample_chunk = changed_chunks[0]
        preview = sample_chunk.body_text.replace("\n", " ").strip()
        if len(preview) > 120:
            preview = preview[:117] + "..."
        report(
            f"Sample chunk: doc={sample_chunk.doc_id} index={sample_chunk.chunk_index} sha={sample_chunk.chunk_sha256[:12]} preview='{preview}'",
            level=logging.INFO,
        )
        report(
            f"Sample point id (first chunk): {sample_chunk.chunk_id}",
            level=logging.INFO,
        )

        title_texts = [chunk.title_text for chunk in changed_chunks]
        title_shas = [chunk.title_sha256 for chunk in changed_chunks]
        body_texts = [chunk.body_text for chunk in changed_chunks]
        body_shas = [chunk.body_sha256 for chunk in changed_chunks]

        title_embeddings = self.embedding_client.embed_texts(title_texts, title_shas)
        body_embeddings = self.embedding_client.embed_texts(body_texts, body_shas)

        title_vectors: Dict[str, List[float]] = {
            result.sha: result.vector for result in title_embeddings
        }
        body_vectors: Dict[str, List[float]] = {
            result.sha: result.vector for result in body_embeddings
        }

        vector_dim = len(next(iter(body_vectors.values()))) if body_vectors else 0
        expected_dim = self.vector_store.vector_size_for_model(embedding_model)
        if body_vectors and vector_dim != expected_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {expected_dim} from {embedding_model} "
                f"but received {vector_dim}"
            )
        report(
            f"[UPSERT] collection={self.vector_store.collection_name} points={len(changed_chunks)} dim={vector_dim}",
            level=logging.INFO,
        )
        logger.debug(
            "Preparing %d point(s) for collection %s on %s",
            len(changed_chunks),
            self.vector_store.collection_name,
            self.vector_store.endpoint_url,
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
                changed_chunks,
                title_vectors=title_vectors,
                body_vectors=body_vectors,
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
                if delta == 0:
                    report(
                        "[QDRANT] No new points detected. Existing vectors already match the ingested content."
                        " Use 'Rebuild' to force a clean re-index if this is unexpected.",
                        level=logging.INFO,
                    )
            report(
                f"[QDRANT] collection '{self.vector_store.collection_name}' at {self.vector_store.endpoint_url} ready with {total_points} point(s)",
                level=logging.INFO,
            )

        if changed_chunks and body_vectors:
            try:
                sample_vector = body_vectors[changed_chunks[0].body_sha256]
                verify_hits = self.vector_store.search(
                    vector_name="body_vec",
                    query_vector=sample_vector,
                    limit=1,
                )
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
            files_processed=len(processed_documents),
            chunks_created=len(changed_chunks),
            skipped=skipped,
        )
