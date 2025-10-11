from __future__ import annotations

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


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


@dataclass
class ChunkRecord:
    doc_id: str
    chunk_index: int
    chunk_id: str
    chunk_key: str
    title_text: str
    body_text: str
    hierarchy: Dict[str, object]
    law_meta: Dict[str, object]
    source: Dict[str, object]
    plan_version: str
    parser_version: str
    chunk_sha256: str
    title_sha256: str
    body_sha256: str

