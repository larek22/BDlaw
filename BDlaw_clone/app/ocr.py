"""Helpers for invoking OCR pipelines when PDFs lack extractable text."""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, List

logger = logging.getLogger(__name__)


class OCRUnavailableError(RuntimeError):
    """Raised when OCR tooling is not available in the environment."""


def should_run_ocr(pages: Iterable[str], *, min_chars: int, min_avg_chars: float) -> bool:
    pages = list(pages)
    if not pages:
        return True
    total_chars = sum(len(page.strip()) for page in pages)
    if total_chars >= min_chars:
        return False
    non_empty_lines = 0
    char_sum = 0
    for page in pages:
        for line in page.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            non_empty_lines += 1
            char_sum += len(stripped)
    if non_empty_lines == 0:
        return True
    average = char_sum / non_empty_lines
    return average < min_avg_chars


def run_ocr(
    path: Path,
    *,
    timeout_seconds: int = 180,
    languages: str = "rus+eng",
    cache_dir: Path | None = None,
    max_attempts: int = 2,
) -> List[str]:
    """Execute OCR over *path* and return textual pages.

    The result is cached by file hash when *cache_dir* is provided. Retries are
    attempted for transient failures with exponential backoff.
    """

    cache_file: Path | None = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        digest = _hash_file(path)
        cache_key = f"{digest}_{languages}".encode("utf-8")
        cache_name = hashlib.sha256(cache_key).hexdigest()
        cache_file = cache_dir / f"{cache_name}.json"
        if cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                if isinstance(cached, list):
                    return [str(page) for page in cached]
            except json.JSONDecodeError:
                logger.debug("Ignoring corrupt OCR cache %s", cache_file)

    attempts = max(1, max_attempts)
    delay = 1.5
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            pages = _perform_ocr(path, timeout_seconds=timeout_seconds, languages=languages)
            if cache_file is not None:
                try:
                    cache_file.write_text(json.dumps(pages, ensure_ascii=False), encoding="utf-8")
                except Exception:  # pragma: no cover - cache best effort
                    logger.debug("Failed to persist OCR cache for %s", cache_file)
            return pages
        except OCRUnavailableError as exc:
            last_error = exc
            if attempt >= attempts:
                break
            time.sleep(delay)
            delay = min(delay * 2, 8)
    assert last_error is not None
    raise last_error


def _perform_ocr(path: Path, *, timeout_seconds: int, languages: str) -> List[str]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _run_ocrmypdf(path, timeout_seconds=timeout_seconds, languages=languages)
    return _run_image_ocr(path, timeout_seconds=timeout_seconds, languages=languages)


def _run_ocrmypdf(path: Path, *, timeout_seconds: int, languages: str) -> List[str]:
    with TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        sidecar = tmp_root / "output.txt"
        output_pdf = tmp_root / "output.pdf"
        command = [
            "ocrmypdf",
            "--force-ocr",
            "--rotate-pages",
            "--deskew",
            "--remove-background",
            "--optimize",
            "0",
            "--language",
            languages,
            "--sidecar",
            str(sidecar),
            str(path),
            str(output_pdf),
        ]
        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
            )
        except FileNotFoundError as exc:  # pragma: no cover - tool missing
            raise OCRUnavailableError("ocrmypdf executable not found") from exc
        except subprocess.SubprocessError as exc:  # pragma: no cover - OCR failure
            logger.warning("OCR command failed for %s: %s", path, exc)
            raise OCRUnavailableError(str(exc)) from exc

        if not sidecar.exists():  # pragma: no cover - unexpected
            raise OCRUnavailableError("OCR sidecar output missing")

        text = sidecar.read_text(encoding="utf-8", errors="ignore")
        return [page.strip() for page in text.split("\f") if page.strip()]


def _run_image_ocr(path: Path, *, timeout_seconds: int, languages: str) -> List[str]:
    with TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        output_base = tmp_root / "image_ocr"
        command = [
            "tesseract",
            str(path),
            str(output_base),
            "-l",
            languages,
            "--psm",
            "6",
        ]
        try:
            subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
            )
        except FileNotFoundError as exc:  # pragma: no cover - tool missing
            raise OCRUnavailableError("tesseract executable not found") from exc
        except subprocess.SubprocessError as exc:  # pragma: no cover - OCR failure
            logger.warning("Image OCR command failed for %s: %s", path, exc)
            raise OCRUnavailableError(str(exc)) from exc

        text_file = output_base.with_suffix(".txt")
        if not text_file.exists():  # pragma: no cover - unexpected
            raise OCRUnavailableError("OCR output missing")
        text = text_file.read_text(encoding="utf-8", errors="ignore").strip()
        return [text] if text else []


def _hash_file(path: Path, chunk_size: int = 65536) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["OCRUnavailableError", "run_ocr", "should_run_ocr"]

