"""Extraction helpers for editorial amendments in legal texts."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, List, Tuple

AMENDMENT_RE = re.compile(
    r"\(в\s+ред\.\s+Федерального\s+закона\s+от\s+(?P<date>\d{2}\.\d{2}\.\d{4})\s+№\s*(?P<no>[\w\-А-Яа-я]+)\)",
    re.IGNORECASE,
)


def extract_amendments(text: str) -> Tuple[str, List[Dict[str, str]]]:
    """Return cleaned text and amendment descriptors."""

    amendments: List[Dict[str, str]] = []

    def repl(match: re.Match[str]) -> str:
        iso_date = datetime.strptime(match.group("date"), "%d.%m.%Y").date().isoformat()
        amendments.append({"date": iso_date, "law_no": match.group("no"), "scope": "article"})
        return ""

    cleaned = AMENDMENT_RE.sub(repl, text)
    return cleaned, amendments


__all__ = ["extract_amendments"]
