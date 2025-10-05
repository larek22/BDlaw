"""OCR importer for scanned images and PDFs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import pytesseract
from PIL import Image

from bdlaw.ingest.base import Document, Importer
from bdlaw.settings.config import OCRConfig


class OCRImporter(Importer):
    mime_types: Iterable[str] = (
        "image/png",
        "image/jpeg",
        "image/tiff",
    )
    extensions: Iterable[str] = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

    def __init__(self, config: OCRConfig):
        self.config = config
        if config.tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = config.tesseract_path

    def supports(self, path: Path, mime_type: str) -> bool:
        return path.suffix.lower() in self.extensions or mime_type in self.mime_types

    def extract(self, path: Path) -> Document:
        image = Image.open(path)
        languages = "+".join(self.config.languages)
        text = pytesseract.image_to_string(image, lang=languages)
        metadata: Dict[str, object] = {"ocr_languages": self.config.languages}
        return Document(path=path, mime_type="image/ocr", raw_text=text, metadata=metadata)


__all__ = ["OCRImporter"]
