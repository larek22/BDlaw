from pathlib import Path
from types import SimpleNamespace
from pathlib import Path

import pytest

from app import ocr


def test_run_ocr_uses_cache(monkeypatch, tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    pdf_path.write_bytes(b"pdf")
    cache_dir = tmp_path / "cache"
    calls = []

    class FakeSubprocess:
        PIPE = object()
        SubprocessError = RuntimeError

        def __init__(self, runner):
            self._runner = runner

        def run(self, *args, **kwargs):
            return self._runner(*args, **kwargs)

    def fake_run(command, check, stdout, stderr, timeout):
        calls.append(command)
        sidecar_index = command.index("--sidecar") + 1
        sidecar = Path(command[sidecar_index])
        sidecar.write_text("page1\fpage2", encoding="utf-8")
        return SimpleNamespace()

    monkeypatch.setattr(ocr, "subprocess", FakeSubprocess(fake_run))
    pages_first = ocr.run_ocr(pdf_path, cache_dir=cache_dir, languages="rus", timeout_seconds=10)
    pages_second = ocr.run_ocr(pdf_path, cache_dir=cache_dir, languages="rus", timeout_seconds=10)
    assert pages_first == ["page1", "page2"]
    assert pages_second == pages_first
    assert len(calls) == 1


def test_run_ocr_image(monkeypatch, tmp_path):
    image_path = tmp_path / "scan.png"
    image_path.write_bytes(b"img")
    cache_dir = tmp_path / "cache"

    class FakeSubprocess:
        PIPE = object()
        SubprocessError = RuntimeError

        def __init__(self, runner):
            self._runner = runner

        def run(self, *args, **kwargs):
            return self._runner(*args, **kwargs)

    def fake_run(command, check, stdout, stderr, timeout):
        assert command[0] == "tesseract"
        output_base = Path(command[2])
        output_base.with_suffix(".txt").write_text("image text", encoding="utf-8")
        return SimpleNamespace()

    monkeypatch.setattr(ocr, "subprocess", FakeSubprocess(fake_run))
    pages = ocr.run_ocr(image_path, cache_dir=cache_dir, languages="rus", timeout_seconds=10)
    assert pages == ["image text"]


def test_should_run_ocr_thresholds():
    assert ocr.should_run_ocr([], min_chars=10, min_avg_chars=5.0)
    assert not ocr.should_run_ocr(["a" * 20], min_chars=10, min_avg_chars=1.0)
