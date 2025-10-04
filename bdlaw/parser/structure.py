"""Legal structure parsing for Russian legislation."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Dict, Iterable, List, Optional

from bdlaw.ingest.base import Document

ARTICLE_RE = re.compile(r"^\s*Ст(?:атья|\.)\s+(\d+(?:\.\d+)?)\b", re.IGNORECASE | re.MULTILINE)
PART_RE = re.compile(r"^\s*Ч(?:асть|\.)\s+([IVXLC\d]+)\b", re.IGNORECASE | re.MULTILINE)
POINT_RE = re.compile(r"(Пункт|Подпункт)\s+([\w\)\(\.\-]+)\b", re.IGNORECASE)
DIVISION_RE = re.compile(r"^Подраздел\s+\d+\.\s*(.+)$", re.MULTILINE)
CHAPTER_RE = re.compile(r"^Глава\s+\d+\.\s*(.+)$", re.MULTILINE)
SECTION_RE = re.compile(r"^Раздел\s+\d+\.\s*(.+)$", re.MULTILINE)
AMENDMENT_RE = re.compile(
    r"\(в\s+ред\.\s+Федерального\s+закона\s+от\s+(\d{2}\.\d{2}\.\d{4})\s+№\s*([\d\-ФЗ]+)\)",
    re.IGNORECASE,
)
CC_RULING_RE = re.compile(
    r"\(постановление\s+Конституционного\s+суда\s+РФ\s+от\s+(\d{2}\.\d{2}\.\d{4})\s+№\s*([\d\-П]+)\)",
    re.IGNORECASE,
)
CROSS_REF_RE = re.compile(r"ст\.?\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


@dataclass
class Annotation:
    type: str
    date: str
    number: str
    text: str
    source_url: Optional[str] = None


@dataclass
class Amendment:
    date: str
    law_no: str
    scope: str


@dataclass
class CrossReference:
    law_code: Optional[str]
    article: str
    part: Optional[str]


@dataclass
class Norm:
    gid: str
    jurisdiction: str
    act_type: str
    law_code: str
    law_full_title: str
    article: Optional[str] = None
    part: Optional[str] = None
    point: Optional[str] = None
    subpoint: Optional[str] = None
    division: Optional[str] = None
    chapter: Optional[str] = None
    section: Optional[str] = None
    version_id: Optional[str] = None
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    supersedes: Optional[str] = None
    source_url: Optional[str] = None
    citation: Optional[str] = None
    lang: str = "ru"
    raw_text: str = ""
    clean_text: str = ""
    annotations: List[Annotation] = field(default_factory=list)
    amendments: List[Amendment] = field(default_factory=list)
    cross_refs: List[CrossReference] = field(default_factory=list)
    page_spans: List[List[int]] = field(default_factory=list)
    span_offsets: List[List[int]] = field(default_factory=list)
    hash: Optional[str] = None
    tokens_est: Optional[int] = None
    ingested_at: Optional[str] = None
    file_origin: Optional[str] = None


@dataclass
class ParsedDocument:
    norms: List[Norm]


@dataclass
class ParsingContext:
    jurisdiction: str
    act_type: str
    law_code: str
    law_full_title: str
    version_id: str
    valid_from: str
    valid_to: Optional[str] = None
    supersedes: Optional[str] = None
    source_url: Optional[str] = None


def extract_sections(text: str) -> Dict[str, str]:
    result: Dict[str, str] = {}
    if division := DIVISION_RE.search(text):
        result["division"] = division.group(0).strip()
    if chapter := CHAPTER_RE.search(text):
        result["chapter"] = chapter.group(0).strip()
    if section := SECTION_RE.search(text):
        result["section"] = section.group(0).strip()
    return result


def _compute_gid(**kwargs: Optional[str]) -> str:
    material = "|".join((kwargs.get(key) or "") for key in [
        "jurisdiction",
        "law_code",
        "article",
        "part",
        "point",
        "subpoint",
        "version_id",
    ])
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def parse_document(document: Document, context: ParsingContext) -> ParsedDocument:
    text = document.metadata.get("clean_text") or document.raw_text
    sections = extract_sections(text)
    article_blocks = _split_by_article(text)
    norms: List[Norm] = []
    for article_title, article_text in article_blocks:
        parts = _split_by_part(article_text)
        for part_label, part_text in parts:
            points = _split_by_point(part_text)
            if points:
                for point_label, point_text in points:
                    norm = _build_norm(
                        context,
                        sections,
                        article_title,
                        part_label,
                        point_label,
                        point_text,
                        document,
                    )
                    norms.append(norm)
            else:
                norm = _build_norm(
                    context,
                    sections,
                    article_title,
                    part_label,
                    None,
                    part_text,
                    document,
                )
                norms.append(norm)
    return ParsedDocument(norms=norms)


def _split_by_article(text: str) -> List[tuple[str, str]]:
    matches = list(ARTICLE_RE.finditer(text))
    blocks: List[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        blocks.append((match.group(1), text[start:end].strip()))
    if not blocks:
        blocks.append((None, text.strip()))
    return blocks


def _split_by_part(article_text: str) -> List[tuple[Optional[str], str]]:
    matches = list(PART_RE.finditer(article_text))
    if not matches:
        return [(None, article_text.strip())]
    blocks: List[tuple[Optional[str], str]] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(article_text)
        blocks.append((match.group(1), article_text[start:end].strip()))
    return blocks


def _split_by_point(part_text: str) -> List[tuple[str, str]]:
    matches = list(POINT_RE.finditer(part_text))
    if not matches:
        return []
    blocks: List[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(part_text)
        blocks.append((match.group(2), part_text[start:end].strip()))
    return blocks


def _extract_amendments(text: str) -> List[Amendment]:
    amendments: List[Amendment] = []
    for match in AMENDMENT_RE.finditer(text):
        amendments.append(
            Amendment(
                date=_isoformat(match.group(1)),
                law_no=match.group(2),
                scope="article",
            )
        )
    return amendments


def _extract_annotations(text: str) -> List[Annotation]:
    annotations: List[Annotation] = []
    for match in CC_RULING_RE.finditer(text):
        annotations.append(
            Annotation(
                type="cc_ruling",
                date=_isoformat(match.group(1)),
                number=match.group(2),
                text=match.group(0),
            )
        )
    return annotations


def _extract_cross_refs(text: str, law_code: str) -> List[CrossReference]:
    refs: List[CrossReference] = []
    for match in CROSS_REF_RE.finditer(text):
        refs.append(CrossReference(law_code=law_code, article=match.group(1), part=None))
    return refs


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _estimate_tokens(text: str) -> int:
    return max(1, len(text.split()) * 4 // 3)


def _build_norm(
    context: ParsingContext,
    sections: Dict[str, str],
    article: Optional[str],
    part: Optional[str],
    point: Optional[str],
    text: str,
    document: Document,
) -> Norm:
    clean_text = _clean_text(text)
    amendments = _extract_amendments(text)
    annotations = _extract_annotations(text)
    cross_refs = _extract_cross_refs(text, context.law_code)
    gid = _compute_gid(
        jurisdiction=context.jurisdiction,
        law_code=context.law_code,
        article=article,
        part=part,
        point=point,
        subpoint=None,
        version_id=context.version_id,
    )
    citation_parts = [context.law_code]
    if sections.get("division"):
        citation_parts.append(sections["division"])
    if sections.get("chapter"):
        citation_parts.append(sections["chapter"])
    if article:
        citation_parts.append(f"ст. {article}")
    if part:
        citation_parts.append(f"ч. {part}")
    if point:
        citation_parts.append(f"п. {point}")
    citation = ", ".join(citation_parts)
    citation += f" (ред. {context.version_id})"

    norm = Norm(
        gid=gid,
        jurisdiction=context.jurisdiction,
        act_type=context.act_type,
        law_code=context.law_code,
        law_full_title=context.law_full_title,
        article=article,
        part=part,
        point=point,
        division=sections.get("division"),
        chapter=sections.get("chapter"),
        section=sections.get("section"),
        version_id=context.version_id,
        valid_from=context.valid_from,
        valid_to=context.valid_to,
        supersedes=context.supersedes,
        source_url=context.source_url,
        citation=citation,
        raw_text=text,
        clean_text=clean_text,
        amendments=amendments,
        annotations=annotations,
        cross_refs=cross_refs,
        page_spans=document.metadata.get("page_spans", []),
        span_offsets=[[0, len(text)]],
        hash=hashlib.sha256(clean_text.encode("utf-8")).hexdigest(),
        tokens_est=_estimate_tokens(clean_text),
        ingested_at=datetime.now(UTC).isoformat(),
        file_origin=str(document.path),
    )
    return norm


def _isoformat(date_str: str) -> str:
    day, month, year = date_str.split(".")
    return f"{year}-{month}-{day}"


__all__ = [
    "Annotation",
    "Amendment",
    "CrossReference",
    "Norm",
    "ParsedDocument",
    "ParsingContext",
    "parse_document",
]
