"""Helpers for projecting indices between clean and raw text."""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import List, Tuple


def build_offset_map(raw_text: str, clean_text: str) -> List[int]:
    """Return mapping from clean-text indices to raw-text indices."""

    if not clean_text:
        return []
    matcher = SequenceMatcher(None, raw_text, clean_text, autojunk=False)
    mapping: List[int] = [0] * len(clean_text)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for offset in range(j2 - j1):
                mapping[j1 + offset] = i1 + offset
        elif tag == "replace":
            raw_span = max(1, i2 - i1)
            for offset in range(j2 - j1):
                mapping[j1 + offset] = i1 + min(offset, raw_span - 1)
        elif tag == "insert":
            for offset in range(j2 - j1):
                mapping[j1 + offset] = i1
        elif tag == "delete":
            continue
    for idx in range(1, len(mapping)):
        if mapping[idx] < mapping[idx - 1]:
            mapping[idx] = mapping[idx - 1]
    return mapping


def project_span(span: Tuple[int, int], mapping: List[int], raw_length: int) -> Tuple[int, int]:
    """Project a clean-text span into raw-text coordinates."""

    start, end = span
    if not mapping or raw_length == 0:
        return max(0, start), max(0, end)
    if start >= len(mapping):
        raw_start = raw_length
    else:
        raw_start = mapping[start]
    if end <= 0:
        raw_end = raw_start
    else:
        last_index = min(end - 1, len(mapping) - 1)
        raw_end = mapping[last_index] + 1
    raw_start = max(0, min(raw_start, raw_length))
    raw_end = max(raw_start, min(raw_end, raw_length))
    return raw_start, raw_end


def project_relative_span(
    span: Tuple[int, int], mapping: List[int], raw_length: int
) -> Tuple[int, int]:
    """Project span and clamp to raw-text boundaries returning absolute offsets."""

    raw_start, raw_end = project_span(span, mapping, raw_length)
    return raw_start, raw_end


__all__ = ["build_offset_map", "project_relative_span", "project_span"]
