from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ..settings import AppSettings
    from ..readers.factory import DocumentReaderFactory
    from ..readers.base import DocumentText


def load_txt(
    path: Path,
    reader_factory: "DocumentReaderFactory",
    settings: "AppSettings",
) -> "DocumentText":
    return reader_factory.read(path)
