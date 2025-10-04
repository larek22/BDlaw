"""Helpers for deterministic norm identifiers."""

from __future__ import annotations

import hashlib

__all__ = ["make_gid", "make_hash"]


def make_gid(
    jurisdiction: str,
    law_code: str,
    article: str | None,
    part: str | None,
    point: str | None,
    subpoint: str | None,
    version_id: str,
) -> str:
    material = "|".join(
        value or "" for value in (jurisdiction, law_code, article, part, point, subpoint, version_id)
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def make_hash(clean_text: str) -> str:
    return hashlib.sha256(clean_text.encode("utf-8")).hexdigest()
