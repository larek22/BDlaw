from __future__ import annotations

from pathlib import Path
from typing import List

from docx import Document

from .base import DocumentReader, DocumentReaderError, DocumentText


class DocxReader(DocumentReader):
    supported_suffixes = (".docx",)

    def read(self, path: Path) -> DocumentText:
        if not path.exists():
            raise DocumentReaderError(f"File not found: {path}")

        try:
            document = Document(str(path))
        except Exception as exc:  # pragma: no cover
            raise DocumentReaderError(str(exc)) from exc

        paragraphs = [p.text.strip() for p in document.paragraphs if p.text.strip()]
        if not paragraphs:
            raise DocumentReaderError(f"No text extracted from {path}")

        return DocumentText(doc_id=path.name, path=path, pages=["\n".join(paragraphs)])
