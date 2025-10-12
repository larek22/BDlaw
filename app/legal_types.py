from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class LawMetadata:
    corpus: str
    part_no: int
    law_no: str | None
    enact_date: str | None
    last_amend_date: str | None
    status: str = "active"
    amendments: List[str] = field(default_factory=list)


@dataclass
class HierarchyMetadata:
    section_roman: str | None
    section_title: str | None
    chapter_no: int | None
    chapter_title: str | None
    article_no: str
    article_title: str


@dataclass
class ArticleRecord:
    doc_id: str
    title_text: str
    article_text: str
    hierarchy: HierarchyMetadata
    law_meta: LawMetadata
    source_path: Path
    source_file_name: str
    source_relative_path: str | None
    source_sha256: str


DEFAULT_PLAN_VERSION = "1.0"
DEFAULT_PARSER_VERSION = "1.1.0"


@dataclass
class ChunkRecord:
    """Normalized chunk representation exchanged between pipeline components."""

    doc_id: str
    chunk_index: int
    chunk_id: str = ""
    title_text: str = ""
    body_text: str = ""
    hierarchy: Dict[str, object] = field(default_factory=dict)
    law_meta: Dict[str, object] = field(default_factory=dict)
    source: Dict[str, object] = field(default_factory=dict)
    chunk_key: Optional[str] = None
    plan_version: Optional[str] = None
    parser_version: Optional[str] = None
    chunk_sha256: str = ""
    title_sha256: str = ""
    body_sha256: str = ""

    def __post_init__(self) -> None:  # pragma: no cover - trivial field normalisation
        if not self.chunk_key:
            if self.doc_id:
                self.chunk_key = f"{self.doc_id}#c{self.chunk_index:04d}"
            else:
                self.chunk_key = f"chunk#c{self.chunk_index:04d}"

        if not self.plan_version:
            self.plan_version = DEFAULT_PLAN_VERSION

        if not self.parser_version:
            self.parser_version = DEFAULT_PARSER_VERSION

