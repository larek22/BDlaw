from __future__ import annotations

import zipfile
from pathlib import Path

from bdlaw.ingest.base import ImportPipeline
from bdlaw.ingest.odt import ODTImporter


def _write_odt(path: Path, text: str) -> None:
    content = f"""<?xml version='1.0' encoding='UTF-8'?>
<office:document-content xmlns:office='urn:oasis:names:tc:opendocument:xmlns:office:1.0'
                         xmlns:text='urn:oasis:names:tc:opendocument:xmlns:text:1.0'>
  <office:body>
    <office:text>
      <text:p>{text}</text:p>
    </office:text>
  </office:body>
</office:document-content>
"""
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("mimetype", "application/vnd.oasis.opendocument.text", compress_type=zipfile.ZIP_STORED)
        archive.writestr("content.xml", content)


def test_odt_importer_extracts_text(tmp_path):
    odt_path = tmp_path / "sample.odt"
    _write_odt(odt_path, "Привет мир")

    importer = ODTImporter()
    document = importer.extract(odt_path)

    assert "Привет" in document.raw_text
    assert document.metadata["paragraph_count"] == 1


def test_import_pipeline_uses_odt_importer(tmp_path):
    odt_path = tmp_path / "sample.odt"
    _write_odt(odt_path, "Первый абзац\nВторой абзац")

    pipeline = ImportPipeline(importers=[ODTImporter()], normalizers=[])
    document = pipeline.import_document(odt_path)

    assert "Первый" in document.raw_text
    assert document.mime_type == "application/vnd.oasis.opendocument.text"
