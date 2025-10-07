from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams

from .base import DocumentReader, DocumentReaderError, DocumentText

logger = logging.getLogger(__name__)


class PdfReader(DocumentReader):
    supported_suffixes = (".pdf",)

    def read(self, path: Path) -> DocumentText:
        if not path.exists():
            raise DocumentReaderError(f"File not found: {path}")

        laparams = LAParams()
        pages: List[str] = []
        try:
            with path.open("rb") as f:
                buffer = []
                extract_text_to_fp(
                    f,
                    buffer := _TextBuffer(),
                    laparams=laparams,
                    output_type="text",
                    codec="utf-8",
                )
                pages = buffer.pages
        except Exception as exc:  # pragma: no cover - pdfminer raises many types
            logger.exception("Failed to read PDF %s", path)
            raise DocumentReaderError(str(exc)) from exc

        if not pages:
            raise DocumentReaderError(f"No text extracted from {path}")

        return DocumentText(doc_id=path.name, path=path, pages=pages)


class _TextBuffer:
    def __init__(self) -> None:
        self.pages: List[str] = []
        self._current: List[str] = []

    def write(self, data: str) -> None:
        text = data.replace("\r\n", "\n")
        self._current.append(text)

    def close(self) -> None:
        full_text = "".join(self._current)
        pages = full_text.split("\f")
        self.pages = [page.strip() for page in pages if page.strip()]

    def flush(self) -> None:
        pass
