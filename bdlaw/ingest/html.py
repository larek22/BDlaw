"""HTML importer using readability-lxml and BeautifulSoup."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

from bs4 import BeautifulSoup
from readability import Document as ReadabilityDocument

from bdlaw.ingest.base import Document, Importer


class HTMLImporter(Importer):
    mime_types: Iterable[str] = ("text/html",)
    extensions: Iterable[str] = (".html", ".htm")

    def supports(self, path: Path, mime_type: str) -> bool:
        return path.suffix.lower() in self.extensions or mime_type in self.mime_types

    def extract(self, path: Path) -> Document:
        html = path.read_text(encoding="utf-8")
        readable = ReadabilityDocument(html)
        content_html = readable.summary(html_partial=True)
        soup = BeautifulSoup(content_html, "html.parser")
        text = soup.get_text("\n")
        metadata: Dict[str, object] = {
            "title": readable.title(),
        }
        return Document(path=path, mime_type="text/html", raw_text=text, metadata=metadata)


__all__ = ["HTMLImporter"]
