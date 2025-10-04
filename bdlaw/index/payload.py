"""Pydantic payload schema for indexed legal norms."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, date
from typing import Any, Dict, List, Optional, Tuple

from pydantic import AnyUrl, BaseModel, Field, ConfigDict, field_validator, model_validator


def compute_norm_gid(
    jurisdiction: str,
    law_code: str,
    article: Optional[str],
    part: Optional[str],
    point: Optional[str],
    subpoint: Optional[str],
    version_id: str,
) -> str:
    """Return deterministic gid for the norm."""

    material = "|".join(
        value or "" for value in (jurisdiction, law_code, article, part, point, subpoint, version_id)
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def compute_norm_hash(clean_text: str) -> str:
    """Return stable hash of *clean_text*."""

    return hashlib.sha256(clean_text.encode("utf-8")).hexdigest()


class NormPayload(BaseModel):
    """Validated payload for vector-store ingestion."""

    model_config = ConfigDict(str_strip_whitespace=False, validate_assignment=True, arbitrary_types_allowed=True)

    gid: str
    jurisdiction: str = "ru"
    act_type: str
    division: Optional[str] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    law_code: str
    law_full_title: Optional[str]
    article: str
    part: Optional[str] = None
    point: Optional[str] = None
    subpoint: Optional[str] = None
    version_id: str
    valid_from: date
    valid_to: Optional[date] = None
    supersedes: Optional[str] = None
    source_url: Optional[AnyUrl] = None
    citation: str
    lang: str = "ru"
    raw_text: str
    clean_text: str
    annotations: List[Dict[str, Any]] = Field(default_factory=list)
    amendments: List[Dict[str, Any]] = Field(default_factory=list)
    cross_refs: List[Dict[str, Any]] = Field(default_factory=list)
    page_spans: List[Tuple[int, int]] = Field(default_factory=list)
    span_offsets: List[Tuple[int, int]] = Field(default_factory=list)
    hash: str
    tokens_est: Optional[int] = Field(default=None, ge=0)
    ingested_at: datetime
    file_origin: Optional[str] = None

    @field_validator("gid", "hash")
    @classmethod
    def _validate_hex64(cls, value: str) -> str:
        if len(value) != 64:
            raise ValueError("must be 64 hex chars")
        int(value, 16)
        return value

    @field_validator("page_spans", "span_offsets", mode="before")
    @classmethod
    def _ensure_tuple_list(cls, value: Optional[List[Any]]) -> List[Tuple[int, int]]:
        if value is None:
            return []
        return [tuple(pair) for pair in value]

    @model_validator(mode="after")
    def _finalize(self) -> "NormPayload":
        expected_hash = compute_norm_hash(self.clean_text)
        if self.hash and self.hash != expected_hash:
            raise ValueError("hash does not match clean_text")
        object.__setattr__(self, "hash", expected_hash)
        if not self.gid:
            object.__setattr__(
                self,
                "gid",
                compute_norm_gid(
                    self.jurisdiction,
                    self.law_code,
                    self.article,
                    self.part,
                    self.point,
                    self.subpoint,
                    self.version_id,
                ),
            )
        return self

    @classmethod
    def build(
        cls,
        *,
        jurisdiction: str,
        act_type: str,
        division: Optional[str],
        chapter: Optional[str],
        section: Optional[str],
        law_code: str,
        law_full_title: Optional[str],
        article: str,
        part: Optional[str],
        point: Optional[str],
        subpoint: Optional[str],
        version_id: str,
        valid_from: date,
        valid_to: Optional[date],
        supersedes: Optional[str],
        source_url: Optional[str],
        citation: str,
        lang: str,
        raw_text: str,
        clean_text: str,
        annotations: List[Dict[str, Any]],
        amendments: List[Dict[str, Any]],
        cross_refs: List[Dict[str, Any]],
        page_spans: List[Tuple[int, int]],
        span_offsets: List[Tuple[int, int]],
        tokens_est: Optional[int],
        file_origin: Optional[str],
        ingested_at: Optional[datetime] = None,
    ) -> "NormPayload":
        gid = compute_norm_gid(jurisdiction, law_code, article, part, point, subpoint, version_id)
        return cls(
            gid=gid,
            jurisdiction=jurisdiction,
            act_type=act_type,
            division=division,
            chapter=chapter,
            section=section,
            law_code=law_code,
            law_full_title=law_full_title,
            article=article,
            part=part,
            point=point,
            subpoint=subpoint,
            version_id=version_id,
            valid_from=valid_from,
            valid_to=valid_to,
            supersedes=supersedes,
            source_url=source_url,
            citation=citation,
            lang=lang,
            raw_text=raw_text,
            clean_text=clean_text,
            annotations=annotations,
            amendments=amendments,
            cross_refs=cross_refs,
            page_spans=page_spans,
            span_offsets=span_offsets,
            hash=compute_norm_hash(clean_text),
            tokens_est=tokens_est,
            ingested_at=ingested_at or datetime.now(UTC),
            file_origin=file_origin,
        )


__all__ = ["NormPayload", "compute_norm_gid", "compute_norm_hash"]
