#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import sys
from typing import List, Tuple

from app.data_repository import DataRepository
from app.embeddings import EmbeddingClient
from app.logging_config import configure_logging
from app.qdrant_client import QdrantVectorStore
from app.query_pipeline import QueryPipeline
from app.settings import AppSettings

_REFERENCE_QUERIES = {
    "обязательная доля в наследстве": ["gkrf:part4:art1149"],
    "исключительное право": ["gkrf:part4:art1229", "gkrf:part4:art1255"],
}

_SAMPLE_LIMIT = 5


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


def _check_reference_queries(pipeline: QueryPipeline) -> List[str]:
    failures: List[str] = []
    for question, expected_doc_ids in _REFERENCE_QUERIES.items():
        sources = pipeline.retrieve(question, top_k=5)
        retrieved = [source.chunk.doc_id for source in sources]
        if not retrieved:
            failures.append(f"No results for reference query: {question}")
            continue
        matched = any(
            any(retrieved_doc.startswith(expected) for expected in expected_doc_ids)
            for retrieved_doc in retrieved
        )
        if not matched:
            failures.append(
                f"Reference query '{question}' did not surface {expected_doc_ids}."
                f" Retrieved doc_ids: {retrieved}"
            )
    return failures


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


def main() -> None:
    configure_logging()
    settings = AppSettings.load()

    try:
        vector_store = QdrantVectorStore(settings)
        embedding_client = EmbeddingClient(settings)
        pipeline = QueryPipeline(settings, embedding_client, vector_store)
    except Exception as exc:  # pragma: no cover - runtime failures
        print(f"Verification initialisation failed: {exc}", file=sys.stderr)
        sys.exit(1)

    failures: List[str] = []
    try:
        total_points = vector_store.count_points()
    except Exception as exc:
        failures.append(f"Failed to count Qdrant points: {exc}")
        total_points = 0

    if total_points <= 0:
        failures.append("Qdrant collection is empty. Run ingestion before verification.")

    failures.extend(_check_reference_queries(pipeline))

    repository = DataRepository()
    samples = _load_sample_titles(repository, _SAMPLE_LIMIT)
    if not samples:
        failures.append("No chunk metadata found for self-hit checks.")
    else:
        failures.extend(_check_self_hits(vector_store, embedding_client, samples))

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        sys.exit(1)

    print(
        f"Verification succeeded: {total_points} vectors available, "
        f"{len(_REFERENCE_QUERIES)} reference queries and {len(samples)} self-hits passed."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
