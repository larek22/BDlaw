from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.chunker import chunk_document
from app.readers.base import DocumentText


def test_chunk_document_creates_overlapping_chunks(tmp_path: Path) -> None:
    text = " ".join(["Sentence {}.".format(i) for i in range(1, 30)])
    document = DocumentText(doc_id="sample.txt", path=tmp_path / "sample.txt", pages=[text])

    chunks = chunk_document(document, chunk_size_chars=120, chunk_overlap_chars=30)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text
        assert chunk.sha
    assert chunks[0].doc_id == "sample.txt"
