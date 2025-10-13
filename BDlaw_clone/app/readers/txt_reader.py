from __future__ import annotations

from pathlib import Path

from .base import DocumentReader, DocumentReaderError, DocumentText


class TxtReader(DocumentReader):
    supported_suffixes = (".txt",)

    def read(self, path: Path) -> DocumentText:
        if not path.exists():
            raise DocumentReaderError(f"File not found: {path}")

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:  # pragma: no cover
            raise DocumentReaderError(str(exc)) from exc

        normalized = "\n".join(line.rstrip() for line in text.splitlines())
        if not normalized.strip():
            raise DocumentReaderError(f"No text extracted from {path}")

        return DocumentText(doc_id=path.name, path=path, pages=[normalized])
