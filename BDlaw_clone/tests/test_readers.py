from pathlib import Path
import sys
from pathlib import Path

import pytest

docx = pytest.importorskip("docx")
striprtf = pytest.importorskip("striprtf")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.readers.docx_reader import DocxReader
from app.readers.rtf_reader import RtfReader
from app.readers.txt_reader import TxtReader


Document = docx.Document


def test_txt_reader(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Hello\nWorld", encoding="utf-8")

    reader = TxtReader()
    document = reader.read(file_path)

    assert document.doc_id == "sample.txt"
    assert "Hello" in document.full_text


def test_docx_reader(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.docx"
    doc = Document()
    doc.add_paragraph("Hello from docx")
    doc.save(file_path)

    reader = DocxReader()
    document = reader.read(file_path)

    assert document.doc_id == "sample.docx"
    assert "Hello from docx" in document.full_text


def test_rtf_reader(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.rtf"
    file_path.write_text("{\\rtf1\\ansi Hello RTF}", encoding="utf-8")

    reader = RtfReader()
    document = reader.read(file_path)

    assert document.doc_id == "sample.rtf"
    assert "Hello RTF" in document.full_text
