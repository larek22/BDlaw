from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _normalise(path: Path) -> Path:
    """Return a normalised path with forward slashes for storage."""

    return Path(str(path).replace("\\", "/"))


@dataclass
class RepositoryPaths:
    root: Path
    raw: Path
    staging: Path
    chunks: Path
    snapshots: Path
    logs: Path
    ocr_cache: Path


class DataRepository:
    """Utility for managing on-disk artefacts for the legal corpus pipeline."""

    def __init__(self, root: Path | None = None) -> None:
        self.paths = self._initialise(root or Path("data"))

    def _initialise(self, root: Path) -> RepositoryPaths:
        raw = root / "raw"
        staging = root / "staging"
        chunks = root / "chunks"
        snapshots = root / "snapshots"
        logs = root / "logs"
        ocr_cache = staging / "ocr_cache"
        for directory in (raw, staging, chunks, snapshots, logs, ocr_cache):
            directory.mkdir(parents=True, exist_ok=True)
        return RepositoryPaths(
            root=root,
            raw=raw,
            staging=staging,
            chunks=chunks,
            snapshots=snapshots,
            logs=logs,
            ocr_cache=ocr_cache,
        )

    def derive_slugs(self, source_path: Path) -> tuple[str, str]:
        try:
            relative = source_path.relative_to(self.paths.raw)
            parts = relative.parts
        except ValueError:
            parts = source_path.parts
        corpus_slug = self._slug(parts[0]) if parts else "default"
        part_slug = self._slug(parts[1]) if len(parts) > 1 else "general"
        return corpus_slug, part_slug

    def relative_to_raw(self, source_path: Path) -> str | None:
        try:
            relative = source_path.relative_to(self.paths.raw)
        except ValueError:
            return None
        return _normalise(relative).as_posix()

    def verification_log_path(self) -> Path:
        return self.paths.logs / "verification.log"

    def ensure_subdirectories(self, corpus_slug: str, part_slug: str) -> tuple[Path, Path]:
        staging_dir = self.paths.staging / corpus_slug / part_slug
        chunk_dir = self.paths.chunks / corpus_slug / part_slug
        staging_dir.mkdir(parents=True, exist_ok=True)
        chunk_dir.mkdir(parents=True, exist_ok=True)
        return staging_dir, chunk_dir

    def plan_cache_path(self) -> Path:
        """Return the path used to cache structure plans."""

        return self.paths.staging / "plan_cache.json"

    def router_cache_path(self) -> Path:
        """Return the on-disk cache location for ingestion router plans."""

        return self.paths.staging / "router_cache.sqlite"

    def ocr_cache_path(self) -> Path:
        """Directory for persisted OCR outputs keyed by file hash."""

        self.paths.ocr_cache.mkdir(parents=True, exist_ok=True)
        return self.paths.ocr_cache

    def manifest_path(self, corpus_slug: str, part_slug: str) -> Path:
        staging_dir, _ = self.ensure_subdirectories(corpus_slug, part_slug)
        return staging_dir / f"{part_slug}.manifest.json"

    def normalized_text_path(self, corpus_slug: str, part_slug: str) -> Path:
        staging_dir, _ = self.ensure_subdirectories(corpus_slug, part_slug)
        return staging_dir / f"{part_slug}.normalized.txt"

    def article_json_path(self, corpus_slug: str, part_slug: str) -> Path:
        staging_dir, _ = self.ensure_subdirectories(corpus_slug, part_slug)
        return staging_dir / f"{part_slug}.articles.jsonl"

    def chunk_json_path(self, corpus_slug: str, part_slug: str) -> Path:
        _, chunk_dir = self.ensure_subdirectories(corpus_slug, part_slug)
        return chunk_dir / f"{part_slug}.chunks.jsonl"

    @staticmethod
    def _slug(value: str) -> str:
        value = value.lower()
        value = "".join(ch if ch.isalnum() else "_" for ch in value)
        return value.strip("_") or "default"

