from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ..readers.base import DocumentText

if TYPE_CHECKING:  # pragma: no cover
    from ..readers.factory import DocumentReaderFactory
    from ..settings import AppSettings

logger = logging.getLogger(__name__)


def load_rtf(
    path: Path,
    factory: "DocumentReaderFactory",
    settings: "AppSettings",
) -> DocumentText:
    logger.debug("[LOADER] Reading RTF %s", path)
    return factory.read(path)


__all__ = ["load_rtf"]
