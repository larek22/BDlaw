"""Helpers for splitting semicolon-separated legal enumerations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class ListSegment:
    text: str
    offsets: Tuple[int, int]


def split_semicolon_list(text: str, start_offset: int = 0) -> List[ListSegment]:
    """Return segments for semicolon-separated enumerations."""

    segments: List[ListSegment] = []
    depth = 0
    in_quote = False
    quote_char: str | None = None
    segment_start = 0
    quote_chars = {'"', "'", "«", "»"}
    for idx, char in enumerate(text):
        if char in quote_chars:
            if in_quote and char == quote_char:
                in_quote = False
                quote_char = None
            elif not in_quote:
                in_quote = True
                quote_char = char
        elif char == "(":
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1
        elif char == ";" and depth == 0 and not in_quote:
            remainder = text[idx + 1 :]
            remainder_trim = remainder.lstrip()
            if remainder_trim and (remainder_trim[0].islower() or remainder_trim[0].isdigit()):
                chunk = text[segment_start:idx].strip(" ;\n")
                if chunk:
                    segments.append(
                        ListSegment(
                            text=chunk,
                            offsets=(start_offset + segment_start, start_offset + idx),
                        )
                    )
                segment_start = idx + 1
    tail = text[segment_start:].strip()
    if tail:
        segments.append(
            ListSegment(
                text=tail.strip(" ;\n"),
                offsets=(start_offset + segment_start, start_offset + len(text)),
            )
        )
    return segments if len(segments) > 1 else []


__all__ = ["ListSegment", "split_semicolon_list"]
