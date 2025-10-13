from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

from ..readers.base import DocumentText

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from ..settings import AppSettings
    from ..readers.factory import DocumentReaderFactory

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def load_html(
    path: Path,
    reader_factory: "DocumentReaderFactory",
    settings: "AppSettings",
) -> DocumentText:
    text = path.read_text(encoding="utf-8", errors="ignore")
    stripped = _HTML_TAG_RE.sub(" ", text)
    pages = [line.strip() for line in stripped.splitlines() if line.strip()]
    if not pages:
        pages = [stripped.strip()]
    return DocumentText(doc_id=path.stem, path=path, pages=pages)
