from __future__ import annotations

import logging
from pathlib import Path

from .base import DocumentReader, DocumentReaderError, DocumentText
from ..ocr import OCRUnavailableError, run_ocr
from ..settings import AppSettings
from ..data_repository import DataRepository

logger = logging.getLogger(__name__)


class ImageReader(DocumentReader):
    """Perform OCR over standalone image files to obtain textual pages."""

    supported_suffixes = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

    def __init__(
        self,
        *,
        settings: AppSettings | None = None,
        repository: DataRepository | None = None,
    ) -> None:
        self.settings = settings or AppSettings.load()
        self.repository = repository or DataRepository()

    def read(self, path: Path) -> DocumentText:
        if not path.exists():
            raise DocumentReaderError(f"File not found: {path}")
        if not self.settings.ocr.enabled:
            raise DocumentReaderError("OCR disabled; cannot process image input")
        cache_dir = self.repository.ocr_cache_path()
        try:
            pages = run_ocr(
                path,
                timeout_seconds=self.settings.ocr.timeout_seconds,
                languages=self.settings.ocr.languages,
                cache_dir=cache_dir,
                max_attempts=self.settings.ocr.max_retries,
            )
        except OCRUnavailableError as exc:
            logger.warning("Image OCR unavailable for %s: %s", path, exc)
            raise DocumentReaderError(str(exc)) from exc
        if not pages:
            raise DocumentReaderError(f"OCR produced no text for {path}")
        return DocumentText(doc_id=path.name, path=path, pages=pages)


__all__ = ["ImageReader"]
