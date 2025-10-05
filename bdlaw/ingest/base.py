"""Importer interfaces and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Protocol

from bdlaw.utils.files import guess_mime


@dataclass
class Document:
    path: Path
    mime_type: str
    raw_text: str
    metadata: Dict[str, object]


class Importer(ABC):
    """Base importer interface for document extraction."""

    mime_types: Iterable[str] = ()
    extensions: Iterable[str] = ()

    @abstractmethod
    def supports(self, path: Path, mime_type: str) -> bool:  # pragma: no cover - interface
        """Return ``True`` if this importer can process *path*."""

    @abstractmethod
    def extract(self, path: Path) -> Document:  # pragma: no cover - interface
        """Extract structured :class:`Document` from *path*."""


class Normalizer(Protocol):
    """Protocol for text normalization functions."""

    def __call__(self, document: Document) -> Document:
        ...


class ImportPipeline:
    """Coordinates importer lookup and post-processing."""

    def __init__(self, importers: Iterable[Importer], normalizers: Iterable[Normalizer]):
        self._importers: List[Importer] = list(importers)
        self._normalizers: List[Normalizer] = list(normalizers)

    def import_document(self, path: Path) -> Document:
        mime_type = guess_mime(path)
        for importer in self._importers:
            if importer.supports(path, mime_type):
                document = importer.extract(path)
                for normalizer in self._normalizers:
                    document = normalizer(document)
                return document
        raise ValueError(f"No importer registered for {path}")


__all__ = ["Document", "ImportPipeline", "Importer", "Normalizer"]
