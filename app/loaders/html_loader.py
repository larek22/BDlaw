from __future__ import annotations

from pathlib import Path

from ..readers.base import DocumentText


def load_html(path: Path) -> DocumentText:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return DocumentText(doc_id=path.stem, path=path, pages=[text])


__all__ = ["load_html"]
