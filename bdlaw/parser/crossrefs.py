"""Extraction of cross references from legal norms."""

from __future__ import annotations

import re
from typing import Dict, List, Optional

CROSS_REF_RE = re.compile(
    r"(?:см\.|по)?\s*ст(?:атья|\.)?\s+(?P<article>\d+(?:\.\d+)?)(?:\s*,?\s*ч(?:асть|\.)?\s*(?P<part>[IVXLC\d]+))?(?:\s+(?P<law>[А-ЯЁA-Z]{2,}(?:\s+[А-ЯЁA-Z]{2,}){0,3}))?",
    re.IGNORECASE,
)


def extract_cross_refs(text: str, default_law: str) -> List[Dict[str, Optional[str]]]:
    refs: Dict[tuple[str, Optional[str]], Dict[str, Optional[str]]] = {}
    for match in CROSS_REF_RE.finditer(text):
        article = match.group("article")
        part = match.group("part")
        law = match.group("law")
        key = (article, part)
        refs[key] = {
            "law_code": law.strip() if law else default_law,
            "article": article,
            "part": part,
        }
    return list(refs.values())


__all__ = ["extract_cross_refs"]
