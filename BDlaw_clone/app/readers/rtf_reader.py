from __future__ import annotations

from pathlib import Path

from striprtf.striprtf import rtf_to_text

from .base import DocumentReader, DocumentReaderError, DocumentText


class RtfReader(DocumentReader):
    supported_suffixes = (".rtf",)

    def read(self, path: Path) -> DocumentText:
        if not path.exists():
            raise DocumentReaderError(f"File not found: {path}")

        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
            text = rtf_to_text(raw)
        except Exception as exc:  # pragma: no cover
            raise DocumentReaderError(str(exc)) from exc

        normalized = "\n".join(line.rstrip() for line in text.splitlines())
        if not normalized.strip():
            raise DocumentReaderError(f"No text extracted from {path}")

        return DocumentText(doc_id=path.name, path=path, pages=[normalized])
