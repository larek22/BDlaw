from __future__ import annotations

import hashlib
import uuid


_NAMESPACE = uuid.UUID("9ac81d55-4c5a-4d82-81e3-6d6fba0d6bde")


def chunk_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def point_id_from_sha(sha_hex: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, sha_hex))


__all__ = ["chunk_sha", "point_id_from_sha"]
