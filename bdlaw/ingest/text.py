"""Plain text importer."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

from bdlaw.ingest.base import Document, Importer


class TextImporter(Importer):
    mime_types: Iterable[str] = ("text/plain",)
    extensions: Iterable[str] = (".txt",)

    def supports(self, path: Path, mime_type: str) -> bool:
        return path.suffix.lower() in self.extensions or mime_type in self.mime_types

    def extract(self, path: Path) -> Document:
        raw_text = path.read_text(encoding="utf-8")
        metadata: Dict[str, object] = {}
        return Document(path=path, mime_type="text/plain", raw_text=raw_text, metadata=metadata)


__all__ = ["TextImporter"]
