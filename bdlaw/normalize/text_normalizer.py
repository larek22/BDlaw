"""High fidelity text normalization for legal documents."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

from bdlaw.ingest.base import Document

from .patterns import HEADER_FOOTER_PATTERNS, REPLACEMENT_RULES
from .offsets import build_offset_map

HYPHEN_BREAK_RE = re.compile(r"(?<=\w)-\s*\n(?=\w)")
MULTISPACE_RE = re.compile(r"[ \t]{2,}")
MULTILINE_RE = re.compile(r"\n{3,}")
STAIRCASE_RE = re.compile(r"\n\s{1,6}(?=\S)")
YO_RE = re.compile("ё")


@dataclass(frozen=True)
class NormalizationResult:
    """Bundle containing all normalization outputs."""

    raw_text: str
    clean_text: str
    stats: Dict[str, int]
    offset_map: List[int]


class TextNormalizer:
    """Callable normalizer compatible with :class:`ImportPipeline`."""

    def __init__(self, *, replace_yo: bool = False):
        self.replace_yo = replace_yo

    def __call__(self, document: Document) -> Document:
        result = normalize_text(document.raw_text, replace_yo=self.replace_yo)
        metadata = dict(document.metadata)
        metadata.update(
            {
                "raw_text": result.raw_text,
                "clean_text": result.clean_text,
                "normalization_stats": result.stats,
                "yo_normalized": self.replace_yo,
                "clean_to_raw_map": result.offset_map,
            }
        )
        return Document(
            path=document.path,
            mime_type=document.mime_type,
            raw_text=result.raw_text,
            metadata=metadata,
        )


def normalize_document(document: Document, *, replace_yo: bool = False) -> Document:
    """Normalize *document* returning a new :class:`Document`."""

    normalizer = TextNormalizer(replace_yo=replace_yo)
    return normalizer(document)


def normalize_text(text: str, *, replace_yo: bool = False) -> NormalizationResult:
    """Return normalized version of *text* and stats."""

    stats: Counter[str] = Counter()
    raw_text = text

    text = unicodedata.normalize("NFC", text)
    stats["unicode_normalized"] += int(text != raw_text)

    for rule in REPLACEMENT_RULES:
        text, count = rule.pattern.subn(rule.replacement, text)
        if count:
            stats[rule.description] += count

    if replace_yo:
        text, count = YO_RE.subn("е", text)
        if count:
            stats["yo_replaced"] += count

    text, count = HYPHEN_BREAK_RE.subn("", text)
    if count:
        stats["hyphen_joins"] += count

    text, count = MULTISPACE_RE.subn(" ", text)
    if count:
        stats["extra_spaces"] += count

    text, count = STAIRCASE_RE.subn("\n", text)
    if count:
        stats["staircase_removed"] += count

    for idx, pattern in enumerate(HEADER_FOOTER_PATTERNS):
        text, count = pattern.subn("", text)
        if count:
            stats[f"header_footer_{idx}"] += count

    text, count = MULTILINE_RE.subn("\n\n", text)
    if count:
        stats["multiline_collapse"] += count

    clean_text = text.strip()
    offset_map = build_offset_map(raw_text, clean_text)
    return NormalizationResult(
        raw_text=raw_text,
        clean_text=clean_text,
        stats=dict(stats),
        offset_map=offset_map,
    )


__all__ = ["NormalizationResult", "TextNormalizer", "normalize_document", "normalize_text"]
