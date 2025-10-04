"""PDF importer using PyMuPDF with fallbacks."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import fitz  # PyMuPDF

from bdlaw.ingest.base import Document, Importer


class PDFImporter(Importer):
    mime_types: Iterable[str] = ("application/pdf",)
    extensions: Iterable[str] = (".pdf",)

    def supports(self, path: Path, mime_type: str) -> bool:
        return path.suffix.lower() in self.extensions or mime_type in self.mime_types

    def extract(self, path: Path) -> Document:
        doc = fitz.open(path)
        text_parts = []
        page_spans = []
        for index, page in enumerate(doc, start=1):
            text_parts.append(page.get_text("text"))
            page_spans.append([index, index])
        raw_text = "\n".join(text_parts)
        metadata: Dict[str, object] = {"page_spans": page_spans}
        metadata.update({k: v for k, v in doc.metadata.items() if v})
        return Document(path=path, mime_type="application/pdf", raw_text=raw_text, metadata=metadata)


__all__ = ["PDFImporter"]
