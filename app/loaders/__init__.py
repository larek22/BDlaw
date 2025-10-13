"""Unified document loader helpers for optional ingestion pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..data_repository import DataRepository
from ..settings import AppSettings
from ..readers.base import DocumentText
from .docx_loader import load_docx
from .html_loader import load_html
from .pdf_loader import load_pdf
from .rtf_loader import load_rtf
from .txt_loader import load_txt

LoaderFn = Callable[[Path, AppSettings, DataRepository], DocumentText | None]

_LOADERS: dict[str, LoaderFn] = {
    ".pdf": load_pdf,
    ".docx": load_docx,
    ".rtf": load_rtf,
    ".txt": load_txt,
    ".html": load_html,
    ".htm": load_html,
}


def load_document(
    path: Path,
    *,
    settings: AppSettings,
    repository: DataRepository,
) -> DocumentText | None:
    """Return a :class:`DocumentText` via specialised loaders when available."""

    loader = _LOADERS.get(path.suffix.lower())
    if loader is None:
        return None
    return loader(path, settings, repository)


__all__ = ["load_document"]
