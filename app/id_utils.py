"""Utility helpers for generating deterministic identifiers."""

from __future__ import annotations

import uuid


# A fixed namespace UUID so that generated identifiers remain stable across
# processes and environments while still producing valid UUID strings. The
# specific value is arbitrary but must remain constant once published.
_POINT_NAMESPACE = uuid.UUID("5a4d2f02-9f03-5b79-9d58-4af1462b9f0d")


def make_point_id(doc_id: str, chunk_index: int, version: str | None = None) -> str:
    """Return a deterministic UUIDv5 for the given chunk metadata.

    Qdrant accepts either unsigned integers or UUID strings as point
    identifiers. We rely on UUIDv5 so the IDs stay stable between runs while
    still being valid according to the API contract. The ``doc_id`` already
    encodes the legal document version (e.g. ``v2006-12-18``), but an
    additional ``version`` field can be supplied for future-proofing (for
    example, differentiating embeddings generated with a new chunking policy).
    """

    name = f"{doc_id}|{chunk_index}"
    if version:
        name = f"{name}|{version}"
    return str(uuid.uuid5(_POINT_NAMESPACE, name))


__all__ = ["make_point_id"]

