"""Validation helpers for GPT-produced vectorization plans."""
from __future__ import annotations

import re
from typing import Dict, List

DEFAULT_ARTICLE = r"(?i)^статья\s+\d+(?:\.\d+)?\b"
DEFAULT_PART = r"(?i)^часть\s+(?:первая|вторая|третья|четвертая|[ivxlcdm]+|\d+)\b"
DEFAULT_SECTION = r"(?i)^раздел\s+(?:[ivxlcdm]+|\d+)\b"
DEFAULT_CHAPTER = r"(?i)^глава\s+\d+\b"
DEFAULT_PATH_FIELDS = ["part", "section", "chapter", "article", "version_date"]
DEFAULT_EXCLUDES = [
    r"(?im)^\(в ред\.",
    r"(?im)^Принят\s+Государственной\s+Думой",
    r"(?im)^Одобрен\s+Советом\s+Федерации",
    r"(?im)^Оглавление\b",
]


def _ensure_regex(pattern: str | None, fallback: str) -> str:
    if not pattern:
        return fallback
    try:
        re.compile(pattern)
    except re.error:
        return fallback
    return pattern


def validate_and_fix(plan: Dict[str, object]) -> Dict[str, object]:
    """Validate a GPT plan and fill in safe defaults."""

    if not isinstance(plan, dict):
        raise ValueError("Plan must be a dictionary")

    plan.setdefault("doc_type", "txt")

    for required in ("hierarchy_rules", "chunking_policy", "preamble_rules"):
        if required not in plan:
            raise ValueError(f"Plan missing '{required}'")

    hierarchy = plan.setdefault("hierarchy_rules", {})
    heading = hierarchy.setdefault("heading_regex", {})
    if not isinstance(heading, dict):
        heading = {}
        hierarchy["heading_regex"] = heading
    fallback_map = {
        "part": DEFAULT_PART,
        "section": DEFAULT_SECTION,
        "chapter": DEFAULT_CHAPTER,
        "article": DEFAULT_ARTICLE,
    }
    for level in ("part", "section", "chapter", "article"):
        heading[level] = _ensure_regex(
            str(heading.get(level) or ""),
            fallback_map[level],
        )
    hierarchy["version_regex"] = _ensure_regex(
        hierarchy.get("version_regex") if isinstance(hierarchy.get("version_regex"), str) else None,
        r"(?im)от\s+(\d{2}\.\d{2}\.\d{4})",
    )
    path_fields: List[str] = []
    raw_path_fields = hierarchy.get("path_fields")
    if isinstance(raw_path_fields, list):
        path_fields = [str(item) for item in raw_path_fields if str(item)]
    hierarchy["path_fields"] = [field for field in DEFAULT_PATH_FIELDS if field in (path_fields or DEFAULT_PATH_FIELDS)]

    preamble = plan.setdefault("preamble_rules", {})
    preamble.setdefault("drop_before_first_heading", None)
    excludes = preamble.get("exclude_regexes")
    if not isinstance(excludes, list):
        excludes = []
    else:
        excludes = [pattern for pattern in excludes if isinstance(pattern, str)]
    for default_pattern in DEFAULT_EXCLUDES:
        if default_pattern not in excludes:
            excludes.append(default_pattern)
    preamble["exclude_regexes"] = excludes
    preamble.setdefault("max_frontmatter_chars", 8000)

    chunking = plan.setdefault("chunking_policy", {})
    chunking.setdefault("mode", "by_headings")
    split = chunking.get("split_on_headings")
    if not isinstance(split, list):
        split = ["chapter", "article"]
    else:
        split = [str(item) for item in split if str(item)]
    if "article" not in split:
        split.append("article")
    if "chapter" in split:
        split = [value for value in split if value not in {"chapter", "article"}] + ["chapter", "article"]
    chunking["split_on_headings"] = split
    try:
        chunking["max_tokens"] = max(1, int(chunking.get("max_tokens", 900) or 900))
    except (TypeError, ValueError):
        chunking["max_tokens"] = 900
    try:
        overlap_value = int(chunking.get("overlap_tokens", 100) or 100)
    except (TypeError, ValueError):
        overlap_value = 100
    chunking["overlap_tokens"] = max(0, min(overlap_value, chunking["max_tokens"] - 1 if chunking["max_tokens"] > 1 else 0))
    chunking.setdefault("keep_lists_intact", True)
    chunking.setdefault("keep_tables_intact", True)
    chunking.setdefault("merge_short_paragraphs_under_tokens", 50)

    quality = plan.setdefault("quality_checks", {})
    quality.setdefault("min_chunks", 1)
    try:
        quality_max = int(quality.get("max_tokens_per_chunk", chunking["max_tokens"]))
    except (TypeError, ValueError):
        quality_max = chunking["max_tokens"]
    quality["max_tokens_per_chunk"] = min(quality_max, chunking["max_tokens"])
    quality.setdefault("forbid_empty_text", True)
    quality.setdefault("dedupe_near_duplicates", True)

    return plan


__all__ = ["validate_and_fix"]
