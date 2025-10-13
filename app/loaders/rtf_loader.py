from __future__ import annotations

from pathlib import Path

from ..data_repository import DataRepository
from ..readers.base import DocumentText
from ..readers.rtf_reader import RtfReader
from ..settings import AppSettings


def load_rtf(path: Path, settings: AppSettings, repository: DataRepository) -> DocumentText:
    reader = RtfReader()
    return reader.read(path)


__all__ = ["load_rtf"]
