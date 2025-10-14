from __future__ import annotations

import logging
from pathlib import Path

from ..data_repository import DataRepository
from ..readers.base import DocumentText
from ..readers.pdf_reader import PdfReader
from ..settings import AppSettings

logger = logging.getLogger(__name__)


def load_pdf(path: Path, settings: AppSettings, repository: DataRepository) -> DocumentText:
    """Load a PDF document, optionally invoking OCR when configured."""

    reader = PdfReader(settings=settings, repository=repository)
    document = reader.read(path)
    enable_ocr = getattr(settings.ocr, "enable", settings.ocr.enabled)
    if enable_ocr and len(document.full_text.strip()) < settings.ocr.min_text_chars:
        logger.info(
            "OCR fallback requested for %s (text chars=%d)",
            path.name,
            len(document.full_text),
        )
        # TODO: integrate OCR pipeline; placeholder keeps original text intact for now.
    return document


__all__ = ["load_pdf"]
