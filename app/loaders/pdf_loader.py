from __future__ import annotations

import logging
from pathlib import Path

from ..ocr import run_ocr, should_run_ocr
from ..readers.base import DocumentText
from ..readers.pdf_reader import PdfReader
from ..data_repository import DataRepository
from ..settings import AppSettings

logger = logging.getLogger(__name__)


def load_pdf(path: Path, settings: AppSettings, repository: DataRepository) -> DocumentText:
    reader = PdfReader(settings=settings, repository=repository)
    document = reader.read(path)
    if not settings.ocr.enabled:
        return document
    try:
        if should_run_ocr(
            document.pages,
            min_chars=settings.ocr.min_text_chars,
            min_avg_chars=settings.ocr.min_average_chars_per_line,
        ):
            logger.info("OCR fallback triggered for %s", path)
            ocr_pages = run_ocr(
                path,
                timeout_seconds=settings.ocr.timeout_seconds,
                languages=settings.ocr.languages,
                cache_dir=repository.ocr_cache_path(),
                max_attempts=settings.ocr.max_retries,
            )
            if ocr_pages:
                document = DocumentText(doc_id=document.doc_id, path=path, pages=ocr_pages)
    except Exception as exc:  # pragma: no cover - OCR may not be available
        logger.warning("OCR fallback failed for %s: %s", path, exc)
    return document


__all__ = ["load_pdf"]
