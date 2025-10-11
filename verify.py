#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import sys
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import logging

from app.data_repository import DataRepository
from app.embeddings import EmbeddingClient
from app.logging_config import configure_logging
from app.qdrant_client import QdrantVectorStore
from app.query_pipeline import QueryPipeline
from app.settings import AppSettings
from app.legal_pipeline import doc_prefix

logger = logging.getLogger(__name__)

_REFERENCE_QUERIES = {
    "обязательная доля в наследстве": ["gkrf:part4:art1149", "gkrf:part3:art1149"],
    "исключительное право": ["gkrf:part4:art1229", "gkrf:part4:art1255"],
    "лицензионный договор": ["gkrf:part4:art1235", "gkrf:part4:art1236"],
}

_SAMPLE_LIMIT = 5


@dataclass
class VerificationResult:
    """Structured verification output with severity aware messaging."""

    success: bool
    failures: List[str]
    warnings: List[str]
    info: List[str]

    def all_messages(self) -> List[str]:
        """Return ordered messages for human readable logs."""

        messages: List[str] = []
        messages.extend(self.info)
        messages.extend(f"WARNING: {warning}" for warning in self.warnings)
        messages.extend(self.failures)
        return messages


def _record_total_count(total_points: int, expected_points: int) -> tuple[bool, str]:
    """Return match flag and human readable message for total count reconciliation."""

    if total_points != expected_points:
        message = (
            f"Total point count mismatch: expected {expected_points} chunk(s) "
            f"but found {total_points}."
        )
        return False, message
    return True, f"Total point count matches expected chunk total ({expected_points})."


def _load_sample_titles(repository: DataRepository, limit: int) -> List[Tuple[str, str]]:
    samples: List[Tuple[str, str]] = []
    chunk_dir = repository.paths.chunks
    if not chunk_dir.exists():
        return samples
    for path in sorted(chunk_dir.rglob("*.chunks.jsonl")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    data = json.loads(line)
                    if int(data.get("chunk_index", 0)) == 0:
                        doc_id = data.get("doc_id")
                        title = data.get("title_text")
                        if isinstance(doc_id, str) and isinstance(title, str):
                            samples.append((doc_id, title))
                        break
        except Exception:
            continue
        if len(samples) >= limit * 3:
            break
    random.shuffle(samples)
    return samples[:limit]


def _collect_expected_counts(repository: DataRepository) -> dict[str, int]:
    counts: dict[str, int] = {}
    chunk_dir = repository.paths.chunks
    if not chunk_dir.exists():
        return counts
    for path in sorted(chunk_dir.rglob("*.chunks.jsonl")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    data = json.loads(line)
                    doc_id = data.get("doc_id")
                    if not isinstance(doc_id, str):
                        continue
                    prefix = doc_prefix(doc_id)
                    if prefix:
                        counts[prefix] = counts.get(prefix, 0) + 1
        except Exception:
            continue
    return counts


def _check_reference_queries(
    pipeline: QueryPipeline,
    available_doc_ids: Sequence[str],
) -> tuple[List[str], List[str]]:
    failures: List[str] = []
    info_messages: List[str] = []
    info: List[str] = []

    doc_id_prefixes = {doc_id.split(":v", 1)[0] for doc_id in available_doc_ids}

    for question, expected_doc_ids in _REFERENCE_QUERIES.items():
        active_expectations = [
            prefix
            for prefix in expected_doc_ids
            if prefix in doc_id_prefixes
            or any(doc_id.startswith(prefix) for doc_id in available_doc_ids)
        ]

        if not active_expectations:
            info.append(
                "Skipping reference query '"
                + question
                + "' because expected doc_ids "
                + str(expected_doc_ids)
                + " are not present in the collection."
            )
            continue

        sources = pipeline.retrieve(question, top_k=5)
        retrieved = [source.chunk.doc_id for source in sources]
        if not retrieved:
            failures.append(f"No results for reference query: {question}")
            continue

        matched = any(
            any(retrieved_doc.startswith(expected) for expected in active_expectations)
            for retrieved_doc in retrieved
        )
        if not matched:
            failures.append(
                f"Reference query '{question}' did not surface {active_expectations}."
                f" Retrieved doc_ids: {retrieved}"
            )

    return failures, info


def _check_self_hits(
    vector_store: QdrantVectorStore,
    embedding_client: EmbeddingClient,
    samples: List[Tuple[str, str]],
) -> List[str]:
    failures: List[str] = []
    for doc_id, title in samples:
        try:
            query_vector = embedding_client.embed_query(title)
            filter_ = vector_store.build_doc_id_filter([doc_id])
            hits = vector_store.query(
                vector_name="title_vec",
                query_vector=query_vector,
                limit=1,
                filters=filter_,
            )
        except Exception as exc:
            failures.append(f"Self-hit check failed for {doc_id}: {exc}")
            continue
        if not hits:
            failures.append(f"Self-hit check returned no results for {doc_id}")
            continue
        payload = hits[0].payload or {}
        returned_doc_id = payload.get("doc_id")
        if returned_doc_id != doc_id:
            failures.append(
                f"Self-hit mismatch for {doc_id}: got {returned_doc_id} instead"
            )
    return failures


def _write_log(log_path: Path, lines: List[str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}]\n")
        for line in lines:
            handle.write(line)
            handle.write("\n")
        handle.write("\n")


def run_verification(
    settings: AppSettings | None = None,
    *,
    log_path: Path | None = None,
) -> VerificationResult:
    settings = settings or AppSettings.load()
    repository = DataRepository()
    log_destination = log_path or repository.verification_log_path()

    try:
        vector_store = QdrantVectorStore(settings)
        embedding_client = EmbeddingClient(settings)
        pipeline = QueryPipeline(settings, embedding_client, vector_store)
    except Exception as exc:
        lines = [f"Verification initialisation failed: {exc}"]
        _write_log(log_destination, lines)
        return VerificationResult(success=False, failures=lines, warnings=[], info=[])

    failures: List[str] = []
    info_messages: List[str] = []
    warnings: List[str] = []
    try:
        total_points = vector_store.count_points()
    except Exception as exc:
        failures.append(f"Failed to count Qdrant points: {exc}")
        total_points = 0

    if total_points <= 0:
        failures.append("Qdrant collection is empty. Run ingestion before verification.")

    expected_counts = _collect_expected_counts(repository)
    if expected_counts:
        total_expected = sum(expected_counts.values())
        match, message = _record_total_count(total_points, total_expected)
        if match:
            info_messages.append(message)
        else:
            warnings.append(message)
            logger.warning(message)
        for prefix in sorted(expected_counts):
            expected = expected_counts[prefix]
            try:
                actual = vector_store.count_points_with_prefix(prefix)
            except Exception as exc:
                failures.append(f"Failed to count prefix {prefix}: {exc}")
                continue
            if actual != expected:
                failures.append(
                    f"Prefix {prefix} mismatch: expected {expected} chunk(s) but found {actual}."
                )
            else:
                info_messages.append(
                    f"Prefix {prefix} contains {actual} chunk(s) as expected."
                )
    else:
        info_messages.append("No chunk metadata found on disk for reconciliation checks.")

    try:
        available_doc_ids = list(vector_store.iter_doc_ids())
    except Exception as exc:
        failures.append(f"Failed to enumerate Qdrant doc ids: {exc}")
        available_doc_ids = []

    reference_failures, reference_info = _check_reference_queries(pipeline, available_doc_ids)
    failures.extend(reference_failures)
    info_messages.extend(reference_info)

    samples = _load_sample_titles(repository, _SAMPLE_LIMIT)
    if not samples:
        failures.append("No chunk metadata found for self-hit checks.")
    else:
        failures.extend(_check_self_hits(vector_store, embedding_client, samples))

    summary = (
        f"Verification succeeded: {total_points} vectors available, "
        f"{len(_REFERENCE_QUERIES)} reference queries and {len(samples)} self-hits passed."
    )

    if failures:
        log_lines = ["Verification failed:"]
        log_lines.extend(info_messages)
        log_lines.extend(f"WARNING: {warning}" for warning in warnings)
        log_lines.extend(failures)
        _write_log(log_destination, log_lines)
        return VerificationResult(
            success=False,
            failures=failures,
            warnings=warnings,
            info=info_messages,
        )

    log_lines = info_messages + [summary]
    if warnings:
        log_lines.extend(f"WARNING: {warning}" for warning in warnings)
    _write_log(log_destination, log_lines)
    return VerificationResult(
        success=True,
        failures=[],
        warnings=warnings,
        info=info_messages + [summary],
    )


def main() -> None:
    configure_logging()
    result = run_verification()
    if not result.success:
        for failure in result.failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        if result.warnings:
            for warning in result.warnings:
                print(f"WARNING: {warning}")
        sys.exit(1)
    for warning in result.warnings:
        print(f"WARNING: {warning}")
    print("Verification succeeded.")


if __name__ == "__main__":  # pragma: no cover
    main()
