from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RepositoryPaths:
    root: Path
    raw: Path
    staging: Path
    chunks: Path
    snapshots: Path
    logs: Path


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
        for directory in (raw, staging, chunks, snapshots, logs):
            directory.mkdir(parents=True, exist_ok=True)
        return RepositoryPaths(root=root, raw=raw, staging=staging, chunks=chunks, snapshots=snapshots, logs=logs)

    def derive_slugs(self, source_path: Path) -> tuple[str, str]:
        try:
            relative = source_path.relative_to(self.paths.raw)
        except ValueError:
            parts = source_path.parts
        else:
            parts = relative.parts
        corpus_slug = self._slug(parts[0]) if parts else "default"
        part_slug = self._slug(parts[1]) if len(parts) > 1 else "general"
        return corpus_slug, part_slug

    def ensure_subdirectories(self, corpus_slug: str, part_slug: str) -> tuple[Path, Path]:
        staging_dir = self.paths.staging / corpus_slug / part_slug
        chunk_dir = self.paths.chunks / corpus_slug / part_slug
        staging_dir.mkdir(parents=True, exist_ok=True)
        chunk_dir.mkdir(parents=True, exist_ok=True)
        return staging_dir, chunk_dir

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

