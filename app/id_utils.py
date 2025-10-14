"""Utility helpers for generating deterministic identifiers."""

from __future__ import annotations

import uuid


# A fixed namespace UUID so that generated identifiers remain stable across
# processes and environments while still producing valid UUID strings. The
# specific value is arbitrary but must remain constant once published.
_POINT_NAMESPACE = uuid.UUID("5a4d2f02-9f03-5b79-9d58-4af1462b9f0d")


def make_point_id(
    doc_id: str,
    chunk_index: int,
    version: str | None = None,
    chunk_sha: str | None = None,
    *,
    article_no: str | None = None,
) -> str:
    """Return a deterministic UUIDv5 for the given chunk metadata.

    The historical implementation derived identifiers from a ``chunk_sha``
    whenever available. That behaviour caused point identifiers to drift when
    payload hashing logic changed, ultimately leading to silent overwrites in
    Qdrant. The new logic keys strictly off stable document coordinates so the
    same (doc_id, article_no, chunk_index) tuple always yields the same UUID.
    """

    # DEPRECATED: chunk_sha based identifiers are retained for reference only.
    # if chunk_sha:
    #     name = chunk_sha
    # else:
    #     name = f"{doc_id}|{chunk_index}"
    parts: list[str] = [doc_id.strip(), str(chunk_index)]
    if article_no:
        parts.insert(1, article_no.strip())
    if version:
        parts.append(version.strip())
    name = "|".join(part for part in parts if part)
    if not name:
        raise ValueError("Cannot generate point id without coordinates")
    return str(uuid.uuid5(_POINT_NAMESPACE, name))


__all__ = ["make_point_id"]

