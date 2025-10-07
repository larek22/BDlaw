from __future__ import annotations

import os

from qdrant_client import QdrantClient

from app.settings import AppSettings


def main() -> None:
    settings = AppSettings.load()

    def _clean(value: str | None, fallback: str) -> str:
        if value is None:
            return fallback.strip()
        return value.strip()

    url = _clean(os.getenv("QDRANT_URL"), settings.qdrant.url)
    api_key = _clean(os.getenv("QDRANT_API_KEY"), settings.qdrant.api_key)
    collection = _clean(os.getenv("QDRANT_COLLECTION"), settings.qdrant.collection)

    client = QdrantClient(url=url, api_key=api_key or None)

    collections = client.get_collections()
    print("Collections:")
    for info in collections.collections:
        print(f"- {info.name}")

    try:
        count = client.count(collection_name=collection, exact=True).count
    except Exception as exc:  # noqa: BLE001 - debug helper should surface details
        print(f"Count failed for '{collection}': {exc}")
    else:
        print(f"Collection '{collection}' count: {count}")

if __name__ == "__main__":
    main()
