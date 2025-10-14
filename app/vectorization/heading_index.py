"""Utilities for deriving hierarchical metadata from heading blocks."""
from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .models import VectorizationPlan
from .normalizer import Block

_SECTION_RE = re.compile(r"(?im)\bраздел\s+([IVXLCDM]+)(?:[.\s\-–—]+(?P<title>.+))?$")
_PART_RE = re.compile(r"(?im)\bчасть\s+(\d+)(?:[.\s\-–—]+(?P<title>.+))?$")
_CHAPTER_RE = re.compile(r"(?im)\bглава\s+(\d+)(?:[.\s\-–—]+(?P<title>.+))?$")
_ARTICLE_RE = re.compile(r"(?im)\bст(?:атья|\.)\s+(\d+(?:\.\d+)?)")
_ARTICLE_TITLE_RE = re.compile(r"(?im)\bст(?:атья|\.)\s+\d+(?:\.\d+)?(?:\s*[–—\-]\s*|\.\s*)(.+)$")
_RUSSIAN_KEYWORD_RE = re.compile(r"(?i)\b(статья|глава|часть|раздел)\b")


@dataclass(slots=True)
class HeadingEntry:
    kind: str
    start: int
    end: int
    label: str
    identifier: Optional[str]
    identifier_int: Optional[int]
    suffix: Optional[str]
    rubric: Optional[str]
    language: Optional[str] = None


class HeadingIndex:
    """Lookup structure for nearest headings and derived titles."""

    def __init__(self, entries: Dict[str, List[HeadingEntry]], language: Optional[str]) -> None:
        self._entries: Dict[str, List[HeadingEntry]] = {
            kind: sorted(items, key=lambda item: item.start)
            for kind, items in entries.items()
        }
        self._starts: Dict[str, List[int]] = {
            kind: [item.start for item in items] for kind, items in self._entries.items()
        }
        self._language = language

    @property
    def language(self) -> Optional[str]:
        return self._language

    def nearest(self, kind: str, position: int) -> Optional[HeadingEntry]:
        entries = self._entries.get(kind)
        if not entries:
            return None
        starts = self._starts[kind]
        idx = bisect_right(starts, position) - 1
        if idx < 0:
            return None
        return entries[idx]

    def hierarchy_for(self, position: int) -> Dict[str, object]:
        hierarchy: Dict[str, object] = {}
        section = self.nearest("section", position)
        if section:
            hierarchy["section"] = section.label
            if section.identifier:
                hierarchy["section_roman"] = section.identifier
            if section.rubric:
                hierarchy["section_title"] = section.rubric
        part = self.nearest("part", position)
        if part:
            hierarchy["part"] = part.label
            if part.identifier and part.identifier.isdigit():
                hierarchy["part_no"] = int(part.identifier)
            if part.rubric:
                hierarchy["part_title"] = part.rubric
        chapter = self.nearest("chapter", position)
        if chapter:
            hierarchy["chapter"] = chapter.label
            if chapter.identifier and chapter.identifier.isdigit():
                hierarchy["chapter_no"] = int(chapter.identifier)
            if chapter.rubric:
                hierarchy["chapter_title"] = chapter.rubric
        article = self.nearest("article", position)
        if article:
            hierarchy["article"] = article.label
            if article.identifier:
                hierarchy["article_no"] = article.identifier
                hierarchy["article_no_str"] = article.identifier
                head, suffix = _split_article_number(article.identifier)
                if head is not None:
                    hierarchy["article_no_int"] = head
                if suffix:
                    hierarchy["article_suffix"] = suffix
            if article.rubric:
                hierarchy["article_title"] = article.rubric
        return hierarchy

    def title_for(self, position: int, chunk_text: str, *, fallback: str) -> str:
        article = self.nearest("article", position)
        if article:
            if article.identifier:
                prefix = "Статья" if article.language == "ru" else "Article"
                base = f"{prefix} {article.identifier}"
            else:
                base = article.label
            if article.rubric:
                title = f"{base} — {article.rubric}".strip()
            else:
                title = base.strip()
            return _truncate_title(title)
        chapter = self.nearest("chapter", position)
        if chapter:
            if chapter.identifier:
                prefix = "Глава" if chapter.language == "ru" else "Chapter"
                base = f"{prefix} {chapter.identifier}"
            else:
                base = chapter.label
            if chapter.rubric:
                title = f"{base} — {chapter.rubric}".strip()
            else:
                title = base.strip()
            return _truncate_title(title)
        section = self.nearest("section", position)
        if section:
            if section.identifier:
                prefix = "Раздел" if section.language == "ru" else "Section"
                base = f"{prefix} {section.identifier}"
            else:
                base = section.label
            if section.rubric:
                title = f"{base} — {section.rubric}".strip()
            else:
                title = base.strip()
            return _truncate_title(title)
        snippet = chunk_text.strip().splitlines()[0] if chunk_text.strip() else fallback
        return _truncate_title(snippet or fallback)


def build_heading_index(blocks: Sequence[Block], plan: VectorizationPlan) -> HeadingIndex:
    entries: Dict[str, List[HeadingEntry]] = {"section": [], "part": [], "chapter": [], "article": []}
    regex_map: Dict[str, re.Pattern[str]] = {}
    for name, pattern in (plan.hierarchy_rules.heading_regex or {}).items():
        try:
            regex_map[name] = re.compile(pattern)
        except re.error:
            continue
    detected_language: Optional[str] = None

    for block in blocks:
        if block.type != "heading":
            continue
        text = (block.text or "").strip()
        if not text:
            continue
        start = (block.position or {}).get("start", 0)
        end = (block.position or {}).get("end", start + len(text))
        for kind in ("section", "part", "chapter", "article"):
            entry = _make_entry(kind, text, start, end, regex_map.get(kind))
            if entry:
                entries[kind].append(entry)
                if entry.language and not detected_language:
                    detected_language = entry.language
        if not detected_language and _RUSSIAN_KEYWORD_RE.search(text):
            detected_language = "ru"

    return HeadingIndex(entries, detected_language)


def _clean_heading_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = cleaned.strip("-–—.")
    return cleaned.strip()


def _make_entry(
    kind: str,
    text: str,
    start: int,
    end: int,
    pattern: Optional[re.Pattern[str]],
) -> Optional[HeadingEntry]:
    label = _clean_heading_text(text)
    identifier: Optional[str] = None
    identifier_int: Optional[int] = None
    suffix: Optional[str] = None
    rubric: Optional[str] = None
    language: Optional[str] = None

    if kind == "section":
        match = pattern.search(text) if pattern else _SECTION_RE.search(text)
        if not match:
            return None
        roman = match.group(1)
        identifier = roman.upper() if roman else None
        rubric = _clean_heading_text(match.group("title") or "") or None
        language = "ru" if identifier else None
    elif kind == "part":
        match = pattern.search(text) if pattern else _PART_RE.search(text)
        if not match:
            return None
        identifier = match.group(1)
        rubric = _clean_heading_text(match.group("title") or "") or None
        try:
            identifier_int = int(identifier) if identifier else None
        except ValueError:
            identifier_int = None
        language = "ru" if identifier else None
    elif kind == "chapter":
        match = pattern.search(text) if pattern else _CHAPTER_RE.search(text)
        if not match:
            return None
        identifier = match.group(1)
        rubric = _clean_heading_text(match.group("title") or "") or None
        try:
            identifier_int = int(identifier) if identifier else None
        except ValueError:
            identifier_int = None
        language = "ru" if identifier else None
    elif kind == "article":
        match = pattern.search(text) if pattern else _ARTICLE_RE.search(text)
        if not match:
            return None
        if match.groups():
            identifier = match.group(1).strip()
        else:
            alt_match = _ARTICLE_RE.search(text)
            if not alt_match:
                return None
            identifier = alt_match.group(1).strip()
        rubric_match = _ARTICLE_TITLE_RE.search(text)
        if rubric_match:
            rubric = _clean_heading_text(rubric_match.group(1)) or None
        head, suffix = _split_article_number(identifier)
        identifier_int = head
        if suffix:
            suffix = suffix
        language = "ru"
    else:
        return None

    return HeadingEntry(
        kind=kind,
        start=start,
        end=end,
        label=label,
        identifier=identifier,
        identifier_int=identifier_int,
        suffix=suffix,
        rubric=rubric,
        language=language,
    )


def _split_article_number(identifier: Optional[str]) -> tuple[Optional[int], Optional[str]]:
    if not identifier:
        return None, None
    head_part, _, tail = identifier.partition(".")
    try:
        head_int = int(head_part)
    except ValueError:
        head_int = None
    suffix = tail or None
    return head_int, suffix


def _truncate_title(text: str, max_length: int = 120) -> str:
    cleaned = _clean_heading_text(text)
    if len(cleaned) <= max_length:
        return cleaned
    truncated = cleaned[: max_length - 1].rstrip()
    return f"{truncated}…"


__all__ = ["HeadingIndex", "HeadingEntry", "build_heading_index"]
