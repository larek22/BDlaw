from __future__ import annotations

import logging
from pathlib import Path
from typing import List

try:  # pragma: no cover - optional dependency
    import fitz  # type: ignore
except Exception:  # pragma: no cover - PyMuPDF not available
    fitz = None  # type: ignore

from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams

from .base import DocumentReader, DocumentReaderError, DocumentText

logger = logging.getLogger(__name__)


class PdfReader(DocumentReader):
    supported_suffixes = (".pdf",)

    def read(self, path: Path) -> DocumentText:
        if not path.exists():
            raise DocumentReaderError(f"File not found: {path}")

        if fitz is not None:
            try:
                return self._read_with_pymupdf(path)
            except Exception as exc:  # pragma: no cover - fallback path
                logger.debug(
                    "PyMuPDF failed to parse %s (%s); falling back to pdfminer.",
                    path,
                    exc,
                )

        return self._read_with_pdfminer(path)

    def _read_with_pymupdf(self, path: Path) -> DocumentText:
        assert fitz is not None  # for type checkers
        pages: List[str] = []
        try:
            with fitz.open(path) as document:  # type: ignore[attr-defined]
                for page in document:
                    text = page.get_text("text")  # type: ignore[attr-defined]
                    if text:
                        cleaned = text.replace("\r\n", "\n").strip()
                        if cleaned:
                            pages.append(cleaned)
        except Exception as exc:  # pragma: no cover - PyMuPDF specific errors
            raise DocumentReaderError(str(exc)) from exc

        if not pages:
            raise DocumentReaderError(f"No text extracted from {path}")

        return DocumentText(doc_id=path.name, path=path, pages=pages)

    def _read_with_pdfminer(self, path: Path) -> DocumentText:
        laparams = LAParams()
        try:
            with path.open("rb") as handle:
                buffer = _TextBuffer()
                extract_text_to_fp(
                    handle,
                    buffer,
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
