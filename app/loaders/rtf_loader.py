from __future__ import annotations

from pathlib import Path

from ..readers.base import DocumentText
from ..readers.rtf_reader import RtfReader


def load_rtf(path: Path) -> DocumentText:
    reader = RtfReader()
    return reader.read(path)


__all__ = ["load_rtf"]
