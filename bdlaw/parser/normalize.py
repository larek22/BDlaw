"""Text normalization utilities."""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List

from bdlaw.ingest.base import Document

NBSP_PATTERN = re.compile(r"\xa0")
HYPHEN_BREAK_PATTERN = re.compile(r"(\w+)-\n(\w+)")
MULTI_NEWLINE_PATTERN = re.compile(r"\n{3,}")


ARTIFACT_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"^\s*\d+\s*$", re.MULTILINE),
]


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = NBSP_PATTERN.sub(" ", text)
    return text


def fix_hyphenation(text: str) -> str:
    return HYPHEN_BREAK_PATTERN.sub(lambda m: f"{m.group(1)}{m.group(2)}", text)


def strip_artifacts(text: str) -> str:
    cleaned = text
    for pattern in ARTIFACT_PATTERNS:
        cleaned = pattern.sub("\n", cleaned)
    cleaned = MULTI_NEWLINE_PATTERN.sub("\n\n", cleaned)
    return cleaned.strip()


def normalize_document(document: Document) -> Document:
    text = normalize_unicode(document.raw_text)
    text = fix_hyphenation(text)
    clean_text = strip_artifacts(text)
    metadata = dict(document.metadata)
    metadata["raw_text"] = text
    return Document(
        path=document.path,
        mime_type=document.mime_type,
        raw_text=text,
        metadata={**metadata, "clean_text": clean_text},
    )


__all__ = [
    "normalize_document",
    "normalize_unicode",
    "fix_hyphenation",
    "strip_artifacts",
]
