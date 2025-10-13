from __future__ import annotations

from pathlib import Path

from ..data_repository import DataRepository
from ..readers.base import DocumentText
from ..readers.txt_reader import TxtReader
from ..settings import AppSettings


def load_txt(path: Path, settings: AppSettings, repository: DataRepository) -> DocumentText:
    reader = TxtReader()
    return reader.read(path)


__all__ = ["load_txt"]
