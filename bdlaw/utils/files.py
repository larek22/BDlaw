"""File utility helpers."""

from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Iterable, List, Tuple

SUPPORTED_EXTENSIONS: Tuple[str, ...] = (
    ".pdf",
    ".docx",
    ".rtf",
    ".odt",
    ".txt",
    ".html",
    ".htm",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
)


def guess_mime(path: Path) -> str:
    """Return best-effort MIME type for *path*."""

    mime, _ = mimetypes.guess_type(path.name)
    return mime or "application/octet-stream"


def discover_documents(paths: Iterable[Path]) -> List[Path]:
    """Expand directories and filter supported document types."""

    results: List[Path] = []
    for item in paths:
        if item.is_dir():
            for child in sorted(item.rglob("*")):
                if child.is_file() and child.suffix.lower() in SUPPORTED_EXTENSIONS:
                    results.append(child)
        elif item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
            results.append(item)
    return results


__all__ = ["SUPPORTED_EXTENSIONS", "guess_mime", "discover_documents"]
