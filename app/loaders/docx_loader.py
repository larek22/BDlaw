from __future__ import annotations

from pathlib import Path

from ..data_repository import DataRepository
from ..readers.base import DocumentText
from ..readers.docx_reader import DocxReader
from ..settings import AppSettings


def load_docx(path: Path, settings: AppSettings, repository: DataRepository) -> DocumentText:
    reader = DocxReader()
    return reader.read(path)


__all__ = ["load_docx"]
