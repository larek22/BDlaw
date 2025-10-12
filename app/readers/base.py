from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class DocumentText:
    doc_id: str
    path: Path
    pages: List[str]

    @property
    def full_text(self) -> str:
        return "\n".join(self.pages)


class DocumentReaderError(Exception):
    """Raised when a document cannot be read."""


class DocumentReader:
    supported_suffixes: tuple[str, ...] = ()

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self.supported_suffixes

    def read(self, path: Path) -> DocumentText:
        raise NotImplementedError
