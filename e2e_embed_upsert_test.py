from __future__ import annotations

import os
from hashlib import sha256
import sys


def _clean(value: str | None) -> str:
    return value.strip() if value else ""


if "pytest" in sys.modules:  # pragma: no cover - helper script not for pytest collection
    import pytest

    pytest.skip("debug helper script", allow_module_level=True)


from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from app.embeddings import EmbeddingClient
from app.qdrant_client import _vector_size_for_model
from app.settings import AppSettings


def main() -> None:
    settings = AppSettings.load()

    url = _clean(os.getenv("QDRANT_URL")) or settings.qdrant.url
    api_key = _clean(os.getenv("QDRANT_API_KEY")) or settings.qdrant.api_key
    collection = _clean(os.getenv("QDRANT_COLLECTION")) or settings.qdrant.collection
    embed_model = _clean(os.getenv("OPENAI_EMBEDDING_MODEL")) or settings.openai_models.embedding

    if not url:
        raise SystemExit("QDRANT_URL is not configured")

    client = QdrantClient(url=url, api_key=api_key or None)
    embedding_client = EmbeddingClient(settings)

    dim = _vector_size_for_model(embed_model)
    test_collection = f"{collection}_debug"

    print(f"Using Qdrant endpoint: {url}")
    print(f"Recreating debug collection: {test_collection} (dim={dim})")
    client.recreate_collection(
        collection_name=test_collection,
        vectors_config=rest.VectorParams(size=dim, distance=rest.Distance.COSINE),
    )

    text = "vector sanity check"
    sha = sha256(text.encode("utf-8")).hexdigest()
    vector = embedding_client.embed_texts([text], [sha])[0].vector

    print("Upserting single debug point...")
    client.upsert(
        collection_name=test_collection,
        wait=True,
        points=[
            rest.PointStruct(
                id=1,
                vector=vector,
                payload={"text": text},
            )
        ],
    )

    count = client.count(collection_name=test_collection, exact=True).count
    print(f"Count after upsert: {count}")
    if count != 1:
        raise SystemExit("Debug collection count did not reach 1")

    print("Performing retrieval sanity check...")
    results = client.search(
        collection_name=test_collection,
        query_vector=vector,
        limit=3,
        with_payload=True,
    )
    print(f"Top hits returned: {len(results)}")
    if not results:
        raise SystemExit("Retrieval sanity check returned no results")
    for idx, hit in enumerate(results, start=1):
        print(f"  [{idx}] id={hit.id} score={hit.score:.4f}")


if __name__ == "__main__":
    main()
