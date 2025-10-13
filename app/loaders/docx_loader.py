from __future__ import annotations

from pathlib import Path

from ..readers.base import DocumentText
from ..readers.docx_reader import DocxReader


def load_docx(path: Path) -> DocumentText:
    reader = DocxReader()
    return reader.read(path)


__all__ = ["load_docx"]
