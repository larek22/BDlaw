"""Data models for vectorization planning without external dependencies."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Literal, Optional

DistanceMetric = Literal["cosine", "dot", "euclid"]
PlanSource = Literal["gpt", "fallback", "preset"]
ChunkingMode = Literal["by_headings", "by_paragraphs", "by_records"]


@dataclass(slots=True)
class IdStrategy:
    pattern: str
    ensure_uuid_if_missing: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.pattern, str) or not self.pattern:
            raise ValueError("pattern must be a non-empty string")


@dataclass(slots=True)
class HierarchyRules:
    heading_regex: Dict[str, str] = field(
        default_factory=lambda: {
            "section": r"(?im)^section\s+\d+",  # generic western legal numbering
            "part": r"(?im)^part\s+[ivxcl]+",  # roman numerals
            "chapter": r"(?im)^chapter\s+\d+",  # chapters
            "article": r"(?im)^(article|art\.)\s+\d+",  # articles/clauses
        }
    )
    version_regex: Optional[str] = None
    path_fields: List[str] = field(
        default_factory=lambda: [
            "corpus",
            "section",
            "part",
            "chapter",
            "article",
            "version",
        ]
    )


@dataclass(slots=True)
class RecordChunking:
    primary_key: Optional[str] = None
    fields_include: List[str] = field(default_factory=lambda: ["*"])
    fields_exclude: List[str] = field(default_factory=list)
    records_per_chunk: int = 1

    def __post_init__(self) -> None:
        if self.records_per_chunk < 1:
            raise ValueError("records_per_chunk must be >= 1")


@dataclass(slots=True)
class ChunkingPolicy:
    mode: ChunkingMode = "by_paragraphs"
    max_tokens: int = 1200
    overlap_tokens: int = 120
    split_on_headings: List[str] = field(default_factory=list)
    keep_lists_intact: bool = True
    keep_tables_intact: bool = True
    merge_short_paragraphs_under_tokens: int = 80
    record_chunking: RecordChunking = field(default_factory=RecordChunking)

    def __post_init__(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.overlap_tokens < 0:
            raise ValueError("overlap_tokens cannot be negative")


@dataclass(slots=True)
class PayloadSchema:
    fields: Dict[str, str] = field(
        default_factory=lambda: {
            "doc_id": "string",
            "source_file": "string",
            "lang": "string|null",
        }
    )


@dataclass(slots=True)
class QualityChecks:
    min_chunks: int = 1
    max_tokens_per_chunk: int = 1400
    forbid_empty_text: bool = True
    dedupe_near_duplicates: bool = True


@dataclass(slots=True)
class VectorizationPlan:
    doc_type: str
    collection_name: str
    embedding_model: str
    distance: DistanceMetric
    plan_source: PlanSource
    id_strategy: IdStrategy
    hierarchy_rules: HierarchyRules = field(default_factory=HierarchyRules)
    chunking_policy: ChunkingPolicy = field(default_factory=ChunkingPolicy)
    payload_schema: PayloadSchema = field(default_factory=PayloadSchema)
    quality_checks: QualityChecks = field(default_factory=QualityChecks)
    test_queries: List[str] = field(default_factory=list)

    def model_dump(self) -> Dict[str, object]:
        return asdict(self)

    def model_copy(self, *, update: Optional[Dict[str, object]] = None) -> "VectorizationPlan":
        data = self.model_dump()
        if update:
            data.update(update)
        return VectorizationPlan.from_dict(data)

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "VectorizationPlan":
        chunking_data = dict(payload.get("chunking_policy", {}) or {})
        record_data = chunking_data.pop("record_chunking", {}) or {}
        hierarchy_data = dict(payload.get("hierarchy_rules", {}) or {})
        version_regex = hierarchy_data.get("version_regex")
        if isinstance(version_regex, str) and version_regex.lower() == "null":
            hierarchy_data["version_regex"] = None
        return cls(
            doc_type=str(payload.get("doc_type", "")),
            collection_name=str(payload.get("collection_name", "")),
            embedding_model=str(payload.get("embedding_model", "")),
            distance=payload.get("distance", "cosine"),  # type: ignore[arg-type]
            plan_source=payload.get("plan_source", "fallback"),  # type: ignore[arg-type]
            id_strategy=IdStrategy(**payload.get("id_strategy", {})),
            hierarchy_rules=HierarchyRules(**hierarchy_data),
            chunking_policy=ChunkingPolicy(
                **chunking_data,
                record_chunking=RecordChunking(**record_data) if record_data else RecordChunking(),
            ),
            payload_schema=PayloadSchema(**payload.get("payload_schema", {})),
            quality_checks=QualityChecks(**payload.get("quality_checks", {})),
            test_queries=list(payload.get("test_queries", []) or []),
        )

    @staticmethod
    def json_schema() -> Dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "doc_type": {"type": "string"},
                "collection_name": {"type": "string"},
                "embedding_model": {"type": "string"},
                "distance": {"enum": ["cosine", "dot", "euclid"]},
                "plan_source": {"enum": ["gpt", "fallback", "preset"]},
                "id_strategy": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "ensure_uuid_if_missing": {"type": "boolean"},
                    },
                    "required": ["pattern"],
                },
                "hierarchy_rules": {
                    "type": "object",
                    "properties": {
                        "heading_regex": {
                            "type": "object",
                            "properties": {
                                "section": {"type": "string"},
                                "part": {"type": "string"},
                                "chapter": {"type": "string"},
                                "article": {"type": "string"},
                            },
                        },
                        "version_regex": {"type": ["string", "null"]},
                        "path_fields": {"type": "array", "items": {"type": "string"}},
                    },
                },
                "chunking_policy": {
                    "type": "object",
                    "properties": {
                        "mode": {"enum": ["by_headings", "by_paragraphs", "by_records"]},
                        "max_tokens": {"type": "integer"},
                        "overlap_tokens": {"type": "integer"},
                        "split_on_headings": {"type": "array", "items": {"type": "string"}},
                        "keep_lists_intact": {"type": "boolean"},
                        "keep_tables_intact": {"type": "boolean"},
                        "merge_short_paragraphs_under_tokens": {"type": "integer"},
                        "record_chunking": {
                            "type": "object",
                            "properties": {
                                "primary_key": {"type": ["string", "null"]},
                                "fields_include": {"type": "array", "items": {"type": "string"}},
                                "fields_exclude": {"type": "array", "items": {"type": "string"}},
                                "records_per_chunk": {"type": "integer", "minimum": 1},
                            },
                        },
                    },
                },
                "payload_schema": {"type": "object"},
                "quality_checks": {
                    "type": "object",
                    "properties": {
                        "min_chunks": {"type": "integer"},
                        "max_tokens_per_chunk": {"type": "integer"},
                        "forbid_empty_text": {"type": "boolean"},
                        "dedupe_near_duplicates": {"type": "boolean"},
                    },
                },
                "test_queries": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "doc_type",
                "collection_name",
                "embedding_model",
                "distance",
                "plan_source",
                "id_strategy",
            ],
        }


@dataclass(slots=True)
class ChunkPreview:
    text: str
    tokens: int
    start: int
    end: int
    meta: Dict[str, object]
    id_example: str


@dataclass(slots=True)
class PlanPreview:
    plan: VectorizationPlan
    chunks: List[ChunkPreview]
    stats: Dict[str, object]


__all__ = [
    "VectorizationPlan",
    "PlanPreview",
    "ChunkPreview",
    "ChunkingPolicy",
    "ChunkingMode",
    "DistanceMetric",
    "PlanSource",
    "IdStrategy",
    "HierarchyRules",
    "RecordChunking",
    "PayloadSchema",
    "QualityChecks",
]
