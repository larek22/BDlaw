"""Document normalization utilities for the vectorization pipeline."""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional


@dataclass(slots=True)
class Block:
    type: str
    text: str
    attrs: Dict[str, object] = field(default_factory=dict)
    path: str = ""
    position: Dict[str, int] = field(default_factory=dict)


_HEADING_RE = re.compile(r"^(?P<prefix>#{1,6}|\s*(?:h[1-6]|section|chapter|article)\b[:\s]*)\s*(?P<text>.+)$", re.IGNORECASE)


def _iter_plain_blocks(content: str, *, path: Path) -> Iterator[Block]:
    offset = 0
    current_path = str(path.resolve())
    for line in content.splitlines():
        line_length = len(line)
        if not line.strip():
            offset += line_length + 1
            continue
        match = _HEADING_RE.match(line.strip())
        if match:
            prefix = match.group("prefix").lower()
            level = 1
            if prefix.startswith("#"):
                level = prefix.count("#")
            elif prefix.startswith("h") and prefix[1:].isdigit():
                level = int(prefix[1:])
            elif "section" in prefix:
                level = 2
            elif "chapter" in prefix:
                level = 2
            elif "article" in prefix:
                level = 3
            yield Block(
                type="heading",
                text=match.group("text").strip(),
                attrs={"level": level},
                path=current_path,
                position={"start": offset, "end": offset + line_length},
            )
        else:
            yield Block(
                type="paragraph",
                text=line.strip(),
                attrs={},
                path=current_path,
                position={"start": offset, "end": offset + line_length},
            )
        offset += line_length + 1


def _normalize_csv(path: Path) -> List[Block]:
    blocks: List[Block] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        for index, row in enumerate(reader):
            blocks.append(
                Block(
                    type="record",
                    text=json.dumps(row, ensure_ascii=False),
                    attrs={"fields": headers, "row_index": index, "data": row},
                    path=str(path.resolve()),
                    position={"start": index, "end": index},
                )
            )
    return blocks


def _normalize_jsonl(path: Path) -> List[Block]:
    blocks: List[Block] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                data = {"raw": line.strip()}
            blocks.append(
                Block(
                    type="record",
                    text=json.dumps(data, ensure_ascii=False),
                    attrs={"row_index": index, "data": data},
                    path=str(path.resolve()),
                    position={"start": index, "end": index},
                )
            )
    return blocks


def normalize_document(path: Path, *, content_override: Optional[str] = None) -> List[Block]:
    """Normalize a document into primitive blocks.

    Parameters
    ----------
    path:
        Source document path.
    content_override:
        Optional raw string content used instead of reading the file.
    """

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _normalize_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return _normalize_jsonl(path)

    if content_override is not None:
        content = content_override
    else:
        content = path.read_text(encoding="utf-8", errors="ignore")

    blocks = list(_iter_plain_blocks(content, path=path))
    if not blocks:
        return [
            Block(
                type="paragraph",
                text="",
                attrs={},
                path=str(path.resolve()),
                position={"start": 0, "end": 0},
            )
        ]
    return blocks


__all__ = ["Block", "normalize_document"]
