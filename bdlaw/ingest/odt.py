"""ODT importer built on the document's XML payload."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable
from xml.etree import ElementTree
from zipfile import ZipFile

from bdlaw.ingest.base import Document, Importer


class ODTImporter(Importer):
    """Extract text from OpenDocument Text files without external binaries."""

    mime_types: Iterable[str] = ("application/vnd.oasis.opendocument.text",)
    extensions: Iterable[str] = (".odt",)

    def supports(self, path: Path, mime_type: str) -> bool:
        return path.suffix.lower() in self.extensions or mime_type in self.mime_types

    def extract(self, path: Path) -> Document:
        with ZipFile(path) as archive:
            with archive.open("content.xml") as handle:
                xml_bytes = handle.read()
        root = ElementTree.fromstring(xml_bytes)
        ns = {
            "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
        }
        paragraphs: list[str] = []
        for node in root.findall(".//text:p", ns):
            text_fragments: list[str] = []
            for child in node.iter():
                if child.text:
                    text_fragments.append(child.text)
                if child.tail:
                    text_fragments.append(child.tail)
            paragraph = "".join(text_fragments).strip()
            if paragraph:
                paragraphs.append(paragraph)
        raw_text = "\n".join(paragraphs)
        metadata: Dict[str, object] = {"paragraph_count": len(paragraphs)}
        return Document(
            path=path,
            mime_type=self.mime_types[0],
            raw_text=raw_text,
            metadata=metadata,
        )


__all__ = ["ODTImporter"]
