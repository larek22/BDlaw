from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Mapping, Sequence

from .data_repository import DataRepository
from .id_utils import make_point_id as _shared_make_point_id
from .legal_pipeline import (
    LegalCorpusBuilder,
    ProcessedDocument,
    doc_prefix,
    normalize_text,
)
from .legal_types import ChunkRecord
from .settings import AppSettings
from .qdrant_client import QdrantVectorStore
from .vectorization.chunker import apply_plan as apply_vector_plan
from .vectorization.models import VectorizationPlan
from .vectorization.normalizer import normalize_document
from .vectorization.service import filter_blocks_for_plan
from .vectorization.tokenization import DEFAULT_TOKENIZER

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .embeddings import EmbeddingClient
    from .readers.factory import DocumentReaderFactory
    from .readers.base import DocumentText

logger = logging.getLogger(__name__)


def make_point_id(doc_id: str, chunk_index: int) -> str:
    """Deterministic point ids derived from document id and chunk index."""

    # DEPRECATED: direct uuid.uuid5(NAMESPACE_URL, f"{doc_id}:{chunk_index}") implementation
    # retained for reference. The shared helper now centralises ID generation.
    return _shared_make_point_id(doc_id, chunk_index)


def expected_chunk_id(chunk: ChunkRecord) -> str:
    """Return the canonical identifier for a ``ChunkRecord``."""

    article_no: str | None = None
    if isinstance(chunk.hierarchy, dict):
        raw_article = chunk.hierarchy.get("article_no")
        if raw_article is not None:
            article_no = str(raw_article)
    version_marker = "|".join(
        part for part in (chunk.plan_version, chunk.parser_version) if part
    ) or None
    return _shared_make_point_id(
        chunk.doc_id,
        chunk.chunk_index,
        version=version_marker,
        article_no=article_no,
    )


@dataclass
class IngestStats:
    files_processed: int
    chunks_created: int
    skipped: int
    alias_swapped: bool = False
    validation_passed: bool = False


class IngestService:
    def __init__(
        self,
        settings: AppSettings,
        embedding_client: "EmbeddingClient",
        vector_store: QdrantVectorStore,
        reader_factory: "DocumentReaderFactory" | None = None,
        repository: DataRepository | None = None,
        *,
        verify_after_ingest: bool = True,
    ) -> None:
        self.settings = settings
        self.embedding_client = embedding_client
        self.vector_store = vector_store
        self.repository = repository or DataRepository()
        self.verify_after_ingest = verify_after_ingest
        if reader_factory is None:
            from .readers.factory import DocumentReaderFactory  # local import to avoid optional deps at import time

            reader_factory = DocumentReaderFactory(
                settings=self.settings,
                repository=self.repository,
            )
        self.reader_factory = reader_factory
        self.builder = LegalCorpusBuilder(self.repository, settings)
        self._loader_map: Dict[str, Callable[[Path], "DocumentText"]] = {}
        if self.settings.ingest.enable_new_loaders:
            self._loader_map = self._build_loader_map()

    def ingest(
        self,
        paths: Sequence[Path],
        recreate: bool = False,
        *,
        approved_plans: Mapping[Path, VectorizationPlan] | None = None,
        progress_cb: Callable[[str], None] | None = None,
    ) -> IngestStats:
        resolved_plans: Dict[Path, VectorizationPlan] = {}
        if approved_plans:
            for plan_path, plan in approved_plans.items():
                resolved_plans[plan_path.resolve()] = plan
        normalised_paths = [path.resolve() for path in paths]
        plan_paths = [path for path in normalised_paths if path in resolved_plans]
        legacy_paths = [path for path in normalised_paths if path not in resolved_plans]

        stats_legacy = IngestStats(files_processed=0, chunks_created=0, skipped=0)
        stats_plan = IngestStats(files_processed=0, chunks_created=0, skipped=0)

        if legacy_paths:
            stats_legacy = self._ingest_legacy(
                legacy_paths,
                recreate=recreate,
                progress_cb=progress_cb,
            )
            recreate = False
        if plan_paths:
            stats_plan = self._ingest_with_approved_plans(
                plan_paths,
                resolved_plans,
                recreate=recreate,
                progress_cb=progress_cb,
            )

        return IngestStats(
            files_processed=stats_legacy.files_processed + stats_plan.files_processed,
            chunks_created=stats_legacy.chunks_created + stats_plan.chunks_created,
            skipped=stats_legacy.skipped + stats_plan.skipped,
            alias_swapped=stats_legacy.alias_swapped or stats_plan.alias_swapped,
            validation_passed=stats_legacy.validation_passed or stats_plan.validation_passed,
        )

    def _commit_pending_manifests(
        self, processed_documents: Sequence[ProcessedDocument]
    ) -> None:
        """Atomically promote pending manifests written during processing."""

        seen: set[Path] = set()
        for processed in processed_documents:
            pending = getattr(processed, "pending_manifest_path", None)
            final = getattr(processed, "manifest_path", None)
            if not pending or not final:
                continue
            if pending in seen:
                continue
            seen.add(pending)
            if not pending.exists():
                continue
            final.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.replace(pending, final)
            except FileNotFoundError:
                logger.debug("Pending manifest %s disappeared before commit", pending)
            else:
                logger.info("Committed manifest %s", final)

    def _cleanup_pending_manifests(
        self, processed_documents: Sequence[ProcessedDocument]
    ) -> None:
        """Remove transient pending manifests when no ingestion occurs."""

        seen: set[Path] = set()
        for processed in processed_documents:
            pending = getattr(processed, "pending_manifest_path", None)
            if not pending or pending in seen:
                continue
            seen.add(pending)
            try:
                pending.unlink()
            except FileNotFoundError:
                continue
            except OSError as exc:  # pragma: no cover - filesystem errors rare in tests
                logger.warning("Failed to remove pending manifest %s: %s", pending, exc)
            else:
                logger.info("Removed stale pending manifest %s", pending)

    def _ingest_with_approved_plans(
        self,
        paths: Sequence[Path],
        plan_lookup: Mapping[Path, VectorizationPlan],
        *,
        recreate: bool,
        progress_cb: Callable[[str], None] | None,
    ) -> IngestStats:
        def report(message: str, *, level: int = logging.INFO) -> None:
            logger.log(level, message)
            if progress_cb:
                progress_cb(message)

        if not paths:
            return IngestStats(files_processed=0, chunks_created=0, skipped=0)

        embedding_model = self.settings.openai_models.embedding
        report(
            f"Applying {len(paths)} approved plan(s) using embedding model {embedding_model}",
            level=logging.INFO,
        )
        self.vector_store.ensure_collection(embedding_model=embedding_model, recreate=recreate)

        files_processed = 0
        chunks_created = 0
        skipped = 0

        for path in paths:
            plan = plan_lookup.get(path)
            if plan is None:
                report(
                    f"No approved plan found for {path.name}; skipping",
                    level=logging.WARNING,
                )
                skipped += 1
                continue
            if plan.embedding_model and plan.embedding_model != embedding_model:
                report(
                    f"Plan embedding model {plan.embedding_model} differs from configured {embedding_model}; using configured model",
                    level=logging.WARNING,
                )

            report(
                f"Applying approved plan (source={plan.plan_source}) for {path.name}",
                level=logging.INFO,
            )
            try:
                blocks = normalize_document(path)
            except Exception as exc:
                message = f"Failed to normalize {path}: {exc}"
                logger.error(message)
                if progress_cb:
                    progress_cb(message)
                skipped += 1
                continue

            filtered_blocks = filter_blocks_for_plan(blocks, plan)
            chunk_result = apply_vector_plan(
                filtered_blocks, plan, tokenizer=DEFAULT_TOKENIZER
            )
            if not chunk_result.chunks:
                report(
                    f"Plan produced no chunks for {path.name}; skipping",
                    level=logging.WARNING,
                )
                skipped += 1
                continue

            plan_payload = plan.model_dump()
            plan_hash = hashlib.sha256(
                json.dumps(plan_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest()

            manifest_dir = Path("data") / "manifests" / self.vector_store.collection_name
            manifest_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = manifest_dir / f"{path.stem}.plan.json"
            try:
                manifest_path.write_text(
                    json.dumps(plan_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except OSError as exc:
                report(
                    f"Failed to persist plan manifest for {path.name}: {exc}",
                    level=logging.WARNING,
                )

            chunk_records: List[ChunkRecord] = []
            title_texts: List[str] = []
            title_shas: List[str] = []
            body_texts: List[str] = []
            body_shas: List[str] = []

            for index, chunk in enumerate(chunk_result.chunks):
                doc_id = str(chunk.meta.get("doc_id") or path.stem or "document").strip()
                if not doc_id:
                    doc_id = path.stem or "document"
                doc_id = doc_id[:512]
                text_body = chunk.text
                title_text = str(chunk.meta.get("title_text") or "").strip()
                if not title_text and text_body:
                    title_text = text_body.splitlines()[0].strip()
                if not title_text:
                    title_text = f"{doc_id} chunk {index + 1}"
                title_text = title_text[:512]

                hierarchy_meta: Dict[str, object] = {}
                if isinstance(chunk.meta.get("hierarchy"), dict):
                    for key, value in chunk.meta["hierarchy"].items():
                        if value is not None:
                            hierarchy_meta[key] = value
                for field in plan.hierarchy_rules.path_fields:
                    value = chunk.meta.get(field)
                    if value is not None:
                        hierarchy_meta.setdefault(field, value)

                row_index = chunk.meta.get("row_index")
                if row_index is not None:
                    hierarchy_meta.setdefault("row_index", row_index)

                body_sha = hashlib.sha256(text_body.encode("utf-8")).hexdigest()
                title_sha = hashlib.sha256(title_text.encode("utf-8")).hexdigest()
                article_no = hierarchy_meta.get("article")
                article_str = str(article_no) if article_no is not None else None

                record = ChunkRecord(
                    doc_id=doc_id,
                    chunk_index=index,
                    chunk_id=make_point_id(doc_id, index, article_no=article_str),
                    title_text=title_text,
                    body_text=text_body,
                    hierarchy=hierarchy_meta,
                    law_meta={
                        "plan_hash": plan_hash,
                        "plan_source": plan.plan_source,
                    },
                    source={
                        "path": str(path),
                        "plan_hash": plan_hash,
                        "plan_source": plan.plan_source,
                        "plan_chunk_id": chunk.meta.get("plan_chunk_id", chunk.chunk_id),
                        "row_index": row_index,
                    },
                    chunk_sha256=body_sha,
                    title_sha256=title_sha,
                    body_sha256=body_sha,
                )
                chunk_records.append(record)
                title_texts.append(title_text)
                title_shas.append(title_sha)
                body_texts.append(text_body)
                body_shas.append(body_sha)

            report(
                f"Embedding {len(chunk_records)} chunk(s) for {path.name}",
                level=logging.INFO,
            )

            title_embeddings = self.embedding_client.embed_texts(title_texts, title_shas)
            body_embeddings = self.embedding_client.embed_texts(body_texts, body_shas)
            title_vectors = {item.sha: item.vector for item in title_embeddings}
            body_vectors = {item.sha: item.vector for item in body_embeddings}

            preview = chunk_records[0].body_text.replace("\n", " ").strip()
            if len(preview) > 120:
                preview = preview[:117] + "..."
            report(
                f"Plan sample: doc={chunk_records[0].doc_id} index=0 sha={chunk_records[0].chunk_sha256[:12]} preview='{preview}'",
                level=logging.INFO,
            )

            self.vector_store.upsert_chunks(
                chunk_records,
                title_vectors=title_vectors,
                body_vectors=body_vectors,
            )

            self.vector_store.upsert_collection_plan(
                collection=self.vector_store.collection_name,
                plan_json=plan_payload,
                plan_hash=plan_hash,
                plan_source=plan.plan_source,
                embedding_model=embedding_model,
            )

            files_processed += 1
            chunks_created += len(chunk_records)

        return IngestStats(
            files_processed=files_processed,
            chunks_created=chunks_created,
            skipped=skipped,
            alias_swapped=False,
            validation_passed=False,
        )

    def _ingest_legacy(
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

        start_time = time.perf_counter()
        chunk_size = self.settings.ingest.chunk_size_chars
        overlap = self.settings.ingest.chunk_overlap_chars
        token_size = self.settings.chunking.max_tokens
        token_overlap = self.settings.chunking.overlap_tokens

        report(
            "Preparing ingestion for {count} file(s) "
            "(chunk_size={char_size} overlap={char_overlap} tokens={token_size} token_overlap={token_overlap})".format(
                count=len(paths),
                char_size=chunk_size,
                char_overlap=overlap,
                token_size=token_size,
                token_overlap=token_overlap,
            ),
            level=logging.INFO,
        )

        skipped = 0
        processed_documents: List[ProcessedDocument] = []

        for path in paths:
            try:
                document = self._load_document(path)
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

        prefix_expected_counts: Dict[str, int] = {}
        doc_expected_counts: Dict[str, int] = {}
        changed_chunks: List[ChunkRecord] = []
        changed_doc_ids: set[str] = set()
        unchanged_doc_ids: set[str] = set()
        for processed in processed_documents:
            changed_doc_ids.update(processed.doc_ids_changed)
            unchanged_doc_ids.update(processed.doc_ids_unchanged)
            changed_set = set(processed.doc_ids_changed)
            for chunk in processed.chunks:
                prefix = doc_prefix(chunk.doc_id)
                if prefix:
                    prefix_expected_counts[prefix] = (
                        prefix_expected_counts.get(prefix, 0) + 1
                    )
                doc_expected_counts[chunk.doc_id] = doc_expected_counts.get(chunk.doc_id, 0) + 1
                if chunk.doc_id in changed_set:
                    changed_chunks.append(chunk)

        total_articles = sum(len(processed.articles) for processed in processed_documents)
        total_chunk_count = sum(len(processed.chunks) for processed in processed_documents)
        report(
            f"Parsed {total_articles} articles, {total_chunk_count} chunks",
            level=logging.INFO,
        )

        forced_reingest = False
        if not changed_chunks:
            if (
                self.settings.ingest.force_reingest_if_empty
                and processed_documents
            ):
                try:
                    active_points = self.vector_store.count_points()
                except Exception as exc:
                    logger.warning(
                        "Unable to check active collection size for re-ingest decision: %s",
                        exc,
                    )
                else:
                    if active_points == 0:
                        forced_reingest = True
                        report(
                            "Active collection empty but manifests unchanged; forcing re-ingest",
                            level=logging.WARNING,
                        )
                        changed_chunks = [
                            chunk
                            for processed in processed_documents
                            for chunk in processed.chunks
                        ]
                        if changed_chunks:
                            changed_doc_ids = set(doc_expected_counts.keys())
                            for chunk in changed_chunks:
                                chunk.chunk_id = expected_chunk_id(chunk)
                            report(
                                f"Force re-ingest will process {len(changed_doc_ids)} document(s) "
                                f"covering {len(changed_chunks)} chunk(s)",
                                level=logging.INFO,
                            )
                        else:
                            forced_reingest = False
            if not forced_reingest:
                self._cleanup_pending_manifests(processed_documents)
                return IngestStats(files_processed=0, chunks_created=0, skipped=skipped)

        embedding_model = self.settings.openai_models.embedding
        report(
            f"Ensuring collection '{self.vector_store.collection_name}' for embedding model {embedding_model}",
            level=logging.INFO,
        )
        self.vector_store.ensure_collection(embedding_model=embedding_model, recreate=recreate)

        if changed_doc_ids:
            if self.settings.ingest.atomic_alias_swap:
                report(
                    "Skipping destructive deletes prior to upsert (atomic alias swap active)",
                    level=logging.INFO,
                )
                # DEPRECATED: legacy destructive delete retained for compatibility below.
            else:
                report(
                    f"Deleting {len(changed_doc_ids)} document id(s) prior to upsert",
                    level=logging.INFO,
                )
                self.vector_store.delete_documents(sorted(changed_doc_ids))

        self.embedding_client.reset_usage()

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

        if self.settings.ingest.atomic_alias_swap:
            stats = self._ingest_atomic_flow(
                report=report,
                processed_documents=processed_documents,
                changed_chunks=changed_chunks,
                title_vectors=title_vectors,
                body_vectors=body_vectors,
                embedding_model=embedding_model,
                skipped=skipped,
                doc_expected_counts=doc_expected_counts,
            )
            usage = self.embedding_client.usage_summary()
            elapsed = time.perf_counter() - start_time if changed_chunks else 0.0
            report(
                "Ingestion summary: files={files} chunks={chunks} skipped={skipped} "
                "embed_requests={requests:.0f} cache_hits={cache:.0f} tokens={tokens:.0f} "
                "est_cost=${cost:.4f} elapsed={elapsed:.2f}s".format(
                    files=stats.files_processed,
                    chunks=stats.chunks_created,
                    skipped=stats.skipped,
                    requests=usage["requests"],
                    cache=usage["cache_hits"],
                    tokens=usage["tokens"],
                    cost=usage["cost"],
                    elapsed=elapsed,
                ),
                level=logging.INFO,
            )
            if (
                self.verify_after_ingest
                and stats.chunks_created > 0
                and stats.alias_swapped
            ):
                self._run_post_ingest_verification(report, progress_cb)
            elif (
                self.verify_after_ingest
                and stats.chunks_created > 0
                and not stats.alias_swapped
            ):
                report(
                    "Skipping post-ingest verification because alias swap did not occur",
                    level=logging.WARNING,
                )
            return stats
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
                verify_hits = self.vector_store.query(
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

        if prefix_expected_counts:
            try:
                actual_counts = {
                    prefix: self.vector_store.count_points_with_prefix(prefix)
                    for prefix in prefix_expected_counts
                }
            except Exception as exc:
                message = f"Failed to reconcile Qdrant counts: {exc}"
                logger.error(message)
                if progress_cb:
                    progress_cb(message)
                raise

            mismatches = [
                (prefix, prefix_expected_counts[prefix], actual_counts.get(prefix, 0))
                for prefix in prefix_expected_counts
                if actual_counts.get(prefix, 0) != prefix_expected_counts[prefix]
            ]

            for prefix in sorted(prefix_expected_counts):
                report(
                    f"[QDRANT] prefix {prefix} expected {prefix_expected_counts[prefix]} chunk(s); "
                    f"found {actual_counts.get(prefix, 0)}",
                    level=logging.INFO,
                )

            if mismatches:
                detail = ", ".join(
                    f"{prefix} expected {expected} found {actual}"
                    for prefix, expected, actual in mismatches
                )
                report(
                    f"[QDRANT] Detected count mismatch ({detail}); purging orphaned points.",
                    level=logging.WARNING,
                )
                try:
                    removed = self.vector_store.purge_orphans(
                        prefixes=sorted(prefix_expected_counts)
                    )
                except Exception:
                    logger.exception("Failed to purge orphaned Qdrant points")
                    raise
                else:
                    report(
                        f"[QDRANT] Purge removed {removed} orphaned point(s)",
                        level=logging.INFO,
                    )
                try:
                    actual_counts = {
                        prefix: self.vector_store.count_points_with_prefix(prefix)
                        for prefix in prefix_expected_counts
                    }
                except Exception as exc:
                    message = f"Failed to reconcile Qdrant counts after purge: {exc}"
                    logger.error(message)
                    if progress_cb:
                        progress_cb(message)
                    raise
                post_mismatches = [
                    (prefix, prefix_expected_counts[prefix], actual_counts.get(prefix, 0))
                    for prefix in prefix_expected_counts
                    if actual_counts.get(prefix, 0) != prefix_expected_counts[prefix]
                ]
                if post_mismatches:
                    detail = ", ".join(
                        f"{prefix} expected {expected} found {actual}"
                        for prefix, expected, actual in post_mismatches
                    )
                    message = f"Qdrant reconciliation failed: {detail}"
                    logger.error(message)
                    if progress_cb:
                        progress_cb(message)
                    raise RuntimeError(message)
                report(
                    "[QDRANT] Reconciliation succeeded after orphan purge.",
                    level=logging.INFO,
                )
            else:
                report(
                    "[QDRANT] Reconciliation OK: chunk counts match expected totals.",
                    level=logging.INFO,
                )

        if doc_expected_counts:
            try:
                actual_doc_counts = self.vector_store.count_points_for_doc_ids(
                    self.vector_store.collection_name,
                    doc_expected_counts.keys()
                )
            except Exception as exc:
                message = f"Failed to count Qdrant points for updated documents: {exc}"
                logger.error(message)
                if progress_cb:
                    progress_cb(message)
                raise
            doc_mismatches = [
                (doc_id, doc_expected_counts[doc_id], actual_doc_counts.get(doc_id, 0))
                for doc_id in doc_expected_counts
                if actual_doc_counts.get(doc_id, 0) != doc_expected_counts[doc_id]
            ]
            if doc_mismatches:
                sample = ", ".join(
                    f"{doc_id} expected {expected} found {actual}"
                    for doc_id, expected, actual in doc_mismatches[:5]
                )
                message = (
                    "Document-level reconciliation failed: "
                    + sample
                    + (" ..." if len(doc_mismatches) > 5 else "")
                )
                logger.error(message)
                if progress_cb:
                    progress_cb(message)
                raise RuntimeError(message)
            report(
                "[QDRANT] Document-level counts match expected chunk totals.",
                level=logging.INFO,
            )

        self._commit_pending_manifests(processed_documents)

        stats = IngestStats(
            files_processed=len(processed_documents),
            chunks_created=len(changed_chunks),
            skipped=skipped,
        )
        usage = self.embedding_client.usage_summary()
        elapsed = time.perf_counter() - start_time if changed_chunks else 0.0
        report(
            "Ingestion summary: files={files} chunks={chunks} skipped={skipped} "
            "embed_requests={requests:.0f} cache_hits={cache:.0f} tokens={tokens:.0f} "
            "est_cost=${cost:.4f} elapsed={elapsed:.2f}s".format(
                files=stats.files_processed,
                chunks=stats.chunks_created,
                skipped=stats.skipped,
                requests=usage["requests"],
                cache=usage["cache_hits"],
                tokens=usage["tokens"],
                cost=usage["cost"],
                elapsed=elapsed,
            ),
            level=logging.INFO,
        )
        if self.verify_after_ingest and stats.chunks_created > 0:
            self._run_post_ingest_verification(report, progress_cb)

        return stats

    def _build_loader_map(self) -> Dict[str, Callable[[Path], "DocumentText"]]:
        from .loaders import docx_loader, html_loader, pdf_loader, rtf_loader, txt_loader

        return {
            ".pdf": lambda path: pdf_loader.load_pdf(
                path,
                settings=self.settings,
                repository=self.repository,
            ),
            ".docx": docx_loader.load_docx,
            ".rtf": rtf_loader.load_rtf,
            ".txt": txt_loader.load_txt,
            ".html": lambda path: html_loader.load_html(path),
            ".htm": lambda path: html_loader.load_html(path),
        }

    def _load_document(self, path: Path):
        loader = self._loader_map.get(path.suffix.lower()) if self._loader_map else None
        if loader:
            logger.info("Using loader for %s", path.name)
            return loader(path)
        return self.reader_factory.read(path)

    def _ingest_atomic_flow(
        self,
        *,
        report: Callable[[str], None],
        processed_documents: Sequence[ProcessedDocument],
        changed_chunks: Sequence[ChunkRecord],
        title_vectors: Dict[str, List[float]],
        body_vectors: Dict[str, List[float]],
        embedding_model: str,
        skipped: int,
        doc_expected_counts: Dict[str, int],
    ) -> IngestStats:
        alias = self.vector_store.alias_name
        expected_dim = self.vector_store.vector_size_for_model(embedding_model)
        target_collection = self.vector_store.ensure_shadow_collection(
            alias,
            expected_dim,
        )
        report(
            f"Atomic staging collection prepared: {target_collection}",
            level=logging.INFO,
        )

        # DEPRECATED: hard failure on duplicate chunk identifiers previously aborted
        # ingestion runs. The pipeline now deduplicates conflicting chunks while
        # preserving deterministic identifiers for idempotent upserts.
        # seen_ids: set[str] = set()
        # for chunk in changed_chunks:
        #     deterministic_id = expected_chunk_id(chunk)
        #     if chunk.chunk_id != deterministic_id:
        #         chunk.chunk_id = deterministic_id
        #     if chunk.chunk_id in seen_ids:
        #         message = (
        #             f"Duplicate chunk id detected before upsert: {chunk.chunk_id}"
        #         )
        #         logger.error(message)
        #         raise ValueError(message)
        #     seen_ids.add(chunk.chunk_id)

        deduped_chunks: list[ChunkRecord] = []
        chunk_index_by_id: dict[str, int] = {}
        identical_duplicates = 0
        replaced_duplicates = 0
        for chunk in changed_chunks:
            deterministic_id = expected_chunk_id(chunk)
            if chunk.chunk_id != deterministic_id:
                chunk.chunk_id = deterministic_id
            existing_idx = chunk_index_by_id.get(chunk.chunk_id)
            if existing_idx is None:
                chunk_index_by_id[chunk.chunk_id] = len(deduped_chunks)
                deduped_chunks.append(chunk)
                continue

            existing = deduped_chunks[existing_idx]
            if (
                existing.body_sha256 == chunk.body_sha256
                and existing.title_sha256 == chunk.title_sha256
                and existing.body_text == chunk.body_text
                and existing.title_text == chunk.title_text
            ):
                identical_duplicates += 1
                logger.warning(
                    "Duplicate chunk id %s matches existing payload; skipping duplicate doc=%s index=%s",
                    chunk.chunk_id,
                    chunk.doc_id,
                    chunk.chunk_index,
                )
                continue

            replaced_duplicates += 1
            deduped_chunks[existing_idx] = chunk
            logger.warning(
                "Duplicate chunk id %s with differing payload detected; replacing doc=%s index=%s with doc=%s index=%s",
                chunk.chunk_id,
                existing.doc_id,
                existing.chunk_index,
                chunk.doc_id,
                chunk.chunk_index,
            )

        if identical_duplicates or replaced_duplicates:
            report(
                "Resolved duplicate chunk ids: identical={identical} replaced={replaced}".format(
                    identical=identical_duplicates,
                    replaced=replaced_duplicates,
                ),
                level=logging.WARNING,
            )

        changed_chunks = deduped_chunks
        if doc_expected_counts:
            deduped_expected: Dict[str, int] = {}
            for chunk in changed_chunks:
                deduped_expected[chunk.doc_id] = deduped_expected.get(chunk.doc_id, 0) + 1
            doc_expected_counts = deduped_expected

        if not changed_chunks:
            report(
                "All candidate chunks were duplicates; skipping Qdrant upsert",
                level=logging.WARNING,
            )
            self._cleanup_pending_manifests(processed_documents)
            return IngestStats(
                files_processed=len(processed_documents),
                chunks_created=0,
                skipped=skipped,
                alias_swapped=False,
                validation_passed=False,
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

        self.vector_store.upsert_chunks(
            changed_chunks,
            title_vectors=title_vectors,
            body_vectors=body_vectors,
            collection_name=target_collection,
        )
        expected_chunks = len(changed_chunks)
        if (
            expected_chunks
            and not self.settings.qdrant.use_wait
            and hasattr(self.vector_store, "wait_for_count")
        ):
            actual_chunks = self.vector_store.wait_for_count(
                target_collection, expected_chunks
            )
        else:
            # DEPRECATED: single immediate count after async upsert retained for compatibility.
            actual_chunks = self.vector_store.count_points(target_collection)
        report(
            f"Validation: expected={expected_chunks} actual={actual_chunks}",
            level=logging.INFO,
        )
        is_count_valid = actual_chunks == expected_chunks
        validation_limit = max(1, self.settings.ingest.validation_sample_k)
        sample_texts = ["Статья 1", "договор", "наследство"]
        health_ok = False
        alias_swapped = False
        validation_passed = False
        if is_count_valid:
            health_ok = self.vector_store.health_probe(
                target_collection,
                sample_texts=sample_texts,
                limit=validation_limit,
            )
            report(
                f"Health probe for {target_collection}: {health_ok}",
                level=logging.INFO,
            )
            validation_passed = health_ok
        else:
            logger.error(
                "Validation count mismatch for %s (expected=%s actual=%s)",
                target_collection,
                expected_chunks,
                actual_chunks,
            )
            try:
                doc_actuals = self.vector_store.count_points_for_doc_ids(
                    target_collection,
                    list(doc_expected_counts.keys()),
                )
            except Exception as exc:
                logger.warning(
                    "Doc-level diagnostics failed for %s: %s",
                    target_collection,
                    exc,
                )
            else:
                shortfalls: list[str] = []
                for doc_id, expected_total in doc_expected_counts.items():
                    actual_total = doc_actuals.get(doc_id, 0)
                    if actual_total < expected_total:
                        shortfalls.append(
                            f"{doc_id}:{actual_total}/{expected_total}"
                        )
                if shortfalls:
                    preview = ", ".join(shortfalls[:10])
                    report(
                        "Validation shortfall details (first {count}): {details}".format(
                            count=min(len(shortfalls), 10),
                            details=preview,
                        ),
                        level=logging.ERROR,
                    )

        if is_count_valid and health_ok:
            if self.settings.ingest.dry_run:
                report(
                    "Dry-run enabled; skipping alias swap despite successful validation",
                    level=logging.INFO,
                )
                report(
                    "Dry-run mode retains pending manifests for manual review",
                    level=logging.INFO,
                )
            else:
                alias_swapped = self.vector_store.swap_alias_atomically(
                    alias, target_collection
                )
                if alias_swapped:
                    report(
                        f"Alias swap OK -> {alias} -> {target_collection}",
                        level=logging.INFO,
                    )
                    self._commit_pending_manifests(processed_documents)
                    self.vector_store.cleanup_old_collections(alias, keep_n=2)
                else:
                    report(
                        "Alias swap did not complete; pending manifests retained for investigation",
                        level=logging.WARNING,
                    )
        else:
            failure_reason = (
                "count mismatch" if not is_count_valid else "health probe failure"
            )
            message = f"Validation failed; alias not swapped ({failure_reason})"
            logger.error(message)
            report(message, level=logging.ERROR)

        return IngestStats(
            files_processed=len(processed_documents),
            chunks_created=len(changed_chunks),
            skipped=skipped,
            alias_swapped=alias_swapped,
            validation_passed=validation_passed,
        )

    def _run_post_ingest_verification(
        self,
        report: Callable[[str], None],
        progress_cb: Callable[[str], None] | None,
    ) -> None:
        try:
            from verify import run_verification  # local import to avoid cycles during packaging

            verification_log = self.repository.verification_log_path()
            result = run_verification(
                self.settings,
                log_path=verification_log,
            )
        except Exception as exc:
            message = f"Post-ingestion verification failed unexpectedly: {exc}"
            logger.error(message)
            if progress_cb:
                progress_cb(message)
            raise
        else:
            prefix = (
                "Post-ingestion verification detected issues:"
                if (result.failures or result.warnings)
                else "Post-ingestion verification completed successfully."
            )
            logger.info(prefix)
            report(prefix)
            if progress_cb:
                progress_cb(prefix)

            for message in result.info:
                logger.info(message)
                if progress_cb:
                    progress_cb(message)

            for warning in result.warnings:
                warning_msg = f"WARNING: {warning}"
                logger.warning(warning_msg)
                if progress_cb:
                    progress_cb(warning_msg)

            if result.failures:
                for failure in result.failures:
                    logger.error(failure)
                    if progress_cb:
                        progress_cb(failure)
                message = (
                    "Verification checks failed after ingestion. See verification log for details."
                )
                logger.error(message)
                if progress_cb:
                    progress_cb(message)
                raise RuntimeError(message)
