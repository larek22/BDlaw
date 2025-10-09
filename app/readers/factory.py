from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from .base import DocumentReader, DocumentReaderError, DocumentText
from .docx_reader import DocxReader
from .pdf_reader import PdfReader
from .rtf_reader import RtfReader
from .txt_reader import TxtReader


class DocumentReaderFactory:
    def __init__(self, readers: Iterable[DocumentReader] | None = None) -> None:
        self._readers: List[DocumentReader] = list(
            readers
            if readers is not None
            else [PdfReader(), DocxReader(), TxtReader(), RtfReader()]
        )

    def get_reader(self, path: Path) -> DocumentReader:
        for reader in self._readers:
            if reader.supports(path):
                return reader
        raise DocumentReaderError(f"Unsupported file type: {path.suffix}")

    def read(self, path: Path) -> DocumentText:
        reader = self.get_reader(path)
        return reader.read(path)
