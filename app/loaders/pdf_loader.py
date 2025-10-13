from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ..ocr import OCRUnavailableError, run_ocr, should_run_ocr

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ..settings import AppSettings
    from ..readers.factory import DocumentReaderFactory
    from ..readers.base import DocumentText

logger = logging.getLogger(__name__)


def load_pdf(
    path: Path,
    reader_factory: "DocumentReaderFactory",
    settings: "AppSettings",
) -> "DocumentText":
    document = reader_factory.read(path)
    if not settings.ocr.enabled:
        return document
    pages = list(getattr(document, "pages", []) or [])
    if not should_run_ocr(
        pages,
        min_chars=settings.ocr.min_text_chars,
        min_avg_chars=settings.ocr.min_average_chars_per_line,
    ):
        return document
    try:
        logger.info("OCR fallback triggered for %s", path.name)
        ocr_pages = run_ocr(
            path,
            timeout_seconds=settings.ocr.timeout_seconds,
            languages=settings.ocr.languages,
        )
    except OCRUnavailableError as exc:
        logger.warning("OCR unavailable for %s: %s", path, exc)
        return document
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("OCR failed for %s: %s", path, exc)
        return document
    if ocr_pages:
        document.pages = [page.strip() for page in ocr_pages if page.strip()]
        logger.info("OCR completed with %d page(s) for %s", len(document.pages), path.name)
    return document
