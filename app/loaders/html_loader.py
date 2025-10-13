from __future__ import annotations

from pathlib import Path

from ..data_repository import DataRepository
from ..readers.base import DocumentText
from ..settings import AppSettings


def load_html(path: Path, settings: AppSettings, repository: DataRepository) -> DocumentText:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return DocumentText(doc_id=path.stem, path=path, pages=[text])


__all__ = ["load_html"]
