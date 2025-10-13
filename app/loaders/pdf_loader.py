from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ..ocr import OCRUnavailableError, run_ocr, should_run_ocr
from ..readers.base import DocumentReaderError, DocumentText

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..readers.factory import DocumentReaderFactory
    from ..settings import AppSettings
    from ..data_repository import DataRepository

logger = logging.getLogger(__name__)


def load_pdf(
    path: Path,
    factory: "DocumentReaderFactory",
    settings: "AppSettings",
    repository: "DataRepository" | None = None,
) -> DocumentText:
    """Load *path* using the PDF reader with optional OCR fallback."""

    reader = factory.get_reader(path)
    document = reader.read(path)
    if not settings.ocr.enabled:
        return document
    try:
        needs_ocr = should_run_ocr(
            document.pages,
            min_chars=settings.ocr.min_text_chars,
            min_avg_chars=settings.ocr.min_average_chars_per_line,
        )
    except Exception:
        needs_ocr = False
    if not needs_ocr:
        return document
    logger.info("[LOADER] %s has sparse text; attempting OCR", path)
    cache_dir = None
    if repository is not None and hasattr(repository, "ocr_cache_path"):
        try:
            cache_dir = repository.ocr_cache_path()
        except Exception:  # pragma: no cover - cache path best effort
            cache_dir = None
    try:
        pages = run_ocr(
            path,
            timeout_seconds=settings.ocr.timeout_seconds,
            languages=settings.ocr.languages,
            cache_dir=cache_dir,
            max_attempts=settings.ocr.max_retries,
        )
    except OCRUnavailableError as exc:
        logger.warning("OCR unavailable for %s: %s", path, exc)
        return document
    except DocumentReaderError as exc:  # pragma: no cover - defensive guard
        logger.warning("PDF reader error during OCR for %s: %s", path, exc)
        return document
    if not pages:
        return document
    return DocumentText(doc_id=document.doc_id, path=path, pages=pages)


__all__ = ["load_pdf"]
