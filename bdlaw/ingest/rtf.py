"""RTF importer via pypandoc."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pypandoc

from bdlaw.ingest.base import Document, Importer


class RTFImporter(Importer):
    mime_types: Iterable[str] = ("application/rtf",)
    extensions: Iterable[str] = (".rtf",)

    def supports(self, path: Path, mime_type: str) -> bool:
        return path.suffix.lower() in self.extensions or mime_type in self.mime_types

    def extract(self, path: Path) -> Document:
        text = pypandoc.convert_file(str(path), "plain")
        metadata: Dict[str, object] = {}
        return Document(path=path, mime_type="text/rtf", raw_text=text, metadata=metadata)


__all__ = ["RTFImporter"]
