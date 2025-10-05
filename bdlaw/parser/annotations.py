"""Extraction of Constitutional Court annotations."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, List, Tuple

ANNOTATION_RE = re.compile(
    r"\((?P<prefix>(?:постановление|определение)\s+Конституционного\s+суда\s+РФ)\s+от\s+(?P<date>\d{2}\.\d{2}\.\d{4})\s+№\s*(?P<no>[\w\-А-Яа-я]+)(?P<tail>[^)]*)\)",
    re.IGNORECASE,
)
URL_RE = re.compile(r"https?://\S+")


def extract_annotations(text: str) -> Tuple[str, List[Dict[str, str]]]:
    annotations: List[Dict[str, str]] = []

    def repl(match: re.Match[str]) -> str:
        iso_date = datetime.strptime(match.group("date"), "%d.%m.%Y").date().isoformat()
        tail = match.group("tail") or ""
        url_match = URL_RE.search(tail)
        annotations.append(
            {
                "type": "cc_ruling",
                "date": iso_date,
                "no": match.group("no"),
                "text": match.group(0),
                "source_url": url_match.group(0) if url_match else None,
            }
        )
        return ""

    cleaned = ANNOTATION_RE.sub(repl, text).strip()
    if cleaned in {".", ",", ";", ""}:
        cleaned = ""
    return cleaned, annotations


__all__ = ["extract_annotations"]
