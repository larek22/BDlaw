from __future__ import annotations

from pathlib import Path

from ..readers.base import DocumentText
from ..readers.txt_reader import TxtReader


def load_txt(path: Path) -> DocumentText:
    reader = TxtReader()
    return reader.read(path)


__all__ = ["load_txt"]
