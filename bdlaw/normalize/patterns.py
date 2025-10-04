"""Reusable regex patterns for stripping artifacts from legal documents."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Pattern

__all__ = [
    "HEADER_FOOTER_PATTERNS",
    "REPLACEMENT_RULES",
    "ReplacementRule",
]


@dataclass(frozen=True)
class ReplacementRule:
    """Describe a regex-based replacement applied during normalization."""

    pattern: Pattern[str]
    replacement: str
    description: str


HEADER_FOOTER_PATTERNS: tuple[Pattern[str], ...] = (
    re.compile(r"^\s*\d+\s*$", re.MULTILINE),
    re.compile(r"^\s*Стр\.\s*\d+\s*$", re.MULTILINE),
    re.compile(r"^\s*\d+\s+из\s+\d+\s*$", re.MULTILINE),
)

REPLACEMENT_RULES: tuple[ReplacementRule, ...] = (
    ReplacementRule(re.compile(r"\xa0"), " ", "nbsp_to_space"),
    ReplacementRule(re.compile(r"[\u201c\u201d\u00ab\u00bb]"), '"', "normalize_quotes"),
    ReplacementRule(re.compile(r"[\u2018\u2019]"), "'", "normalize_apostrophes"),
)
