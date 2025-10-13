#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys

from app.logging_config import configure_logging
from app.settings import AppSettings
from app.qdrant_client import QdrantVectorStore
from app.embeddings import EmbeddingClient
from app.query_pipeline import QueryPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a hybrid search without generating an LLM answer.")
    parser.add_argument("question", help="Natural language query to search for")
    parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to return")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output")
    args = parser.parse_args()

    configure_logging()
    settings = AppSettings.load()

    try:
        vector_store = QdrantVectorStore(settings)
        embedding_client = EmbeddingClient(settings)
        pipeline = QueryPipeline(settings, embedding_client, vector_store)
        sources = pipeline.retrieve(args.question, top_k=args.top_k)
    except Exception as exc:  # pragma: no cover - runtime failures
        print(f"Search failed: {exc}", file=sys.stderr)
        sys.exit(1)

    payload = [
        {
            "rank": idx,
            "doc_id": source.chunk.doc_id,
            "chunk_index": source.chunk.chunk_index,
            "title": source.chunk.title_text,
            "score": source.score,
            "law_meta": source.chunk.law_meta,
            "hierarchy": source.chunk.hierarchy,
        }
        for idx, source in enumerate(sources, start=1)
    ]
    if args.pretty:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":  # pragma: no cover
    main()
