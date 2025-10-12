from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, TYPE_CHECKING

from .base import DocumentReader, DocumentReaderError, DocumentText
from .docx_reader import DocxReader
from .pdf_reader import PdfReader
from .image_reader import ImageReader
from .rtf_reader import RtfReader
from .txt_reader import TxtReader

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ..settings import AppSettings
    from ..data_repository import DataRepository


class DocumentReaderFactory:
    def __init__(
        self,
        readers: Iterable[DocumentReader] | None = None,
        *,
        settings: "AppSettings" | None = None,
        repository: "DataRepository" | None = None,
    ) -> None:
        if readers is not None:
            self._readers = list(readers)
        else:
            from ..settings import AppSettings  # local import to avoid cycle
            from ..data_repository import DataRepository

            resolved = settings or AppSettings.load()
            repo = repository or DataRepository()
            self._readers = [
                PdfReader(settings=resolved, repository=repo),
                ImageReader(settings=resolved, repository=repo),
                DocxReader(),
                TxtReader(),
                RtfReader(),
            ]

    def get_reader(self, path: Path) -> DocumentReader:
        for reader in self._readers:
            if reader.supports(path):
                return reader
        raise DocumentReaderError(f"Unsupported file type: {path.suffix}")

    def read(self, path: Path) -> DocumentText:
        reader = self.get_reader(path)
        return reader.read(path)
