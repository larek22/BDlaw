"""Parse Russian legal texts into structured norm payloads."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from bdlaw.index.schema import NormPayload
from bdlaw.ingest.base import Document
from bdlaw.parser.amendments import extract_amendments
from bdlaw.parser.annotations import extract_annotations
from bdlaw.parser.crossrefs import extract_cross_refs
from bdlaw.parser.list_segmenter import split_semicolon_list
from bdlaw.parser.model import (
    ArticleNode,
    ChapterNode,
    DivisionNode,
    DocumentStructure,
    PartNode,
    PointNode,
    SectionNode,
    SubpointNode,
)

ARTICLE_RE = re.compile(r"^\s*Ст(?:атья|\.)\s+(\d+(?:\.\d+)?)\b", re.IGNORECASE | re.MULTILINE)
PART_RE = re.compile(r"^\s*Ч(?:асть|\.)\s+([IVXLC\d]+)\b", re.IGNORECASE | re.MULTILINE)
POINT_RE = re.compile(r"(?<!\S)(Пункт|Подпункт)\s+([\dа-яА-Яivxlc\)\(\.]+)\b", re.IGNORECASE)
DIVISION_RE = re.compile(r"^Подраздел\s+(?P<num>\d+)\.?\s*(?P<title>.*)$", re.MULTILINE)
CHAPTER_RE = re.compile(r"^Глава\s+(?P<num>\d+)\.?\s*(?P<title>.*)$", re.MULTILINE)
SECTION_RE = re.compile(r"^Раздел\s+(?P<num>\d+)\.?\s*(?P<title>.*)$", re.MULTILINE)
HEADER_STRIP_PATTERNS: Sequence[re.Pattern[str]] = (
    re.compile(r"^Ст(?:атья|\.)\s+\d+(?:\.\d+)?\.?\s*", re.IGNORECASE),
    re.compile(r"^Ч(?:асть|\.)\s+[IVXLC\d]+\.?\s*", re.IGNORECASE),
    re.compile(r"^(Пункт|Подпункт)\s+[\dа-яА-Яivxlc\)\(\.]+\.?\s*", re.IGNORECASE),
)


@dataclass
class TextSlice:
    label: Optional[str]
    start: int
    end: int
    text: str


@dataclass
class PointSlice:
    point: Optional[str]
    subpoint: Optional[str]
    start: int
    end: int
    text: str


@dataclass
class ParsedDocument:
    norms: List[NormPayload]
    structure: DocumentStructure = field(default_factory=DocumentStructure)


@dataclass
class ParsingContext:
    jurisdiction: str
    act_type: str
    law_code: str
    law_full_title: str
    version_id: str
    valid_from: date
    valid_to: Optional[date] = None
    supersedes: Optional[str] = None
    source_url: Optional[str] = None
    lang: str = "ru"


def parse_document(document: Document, context: ParsingContext) -> ParsedDocument:
    """Parse *document* into :class:`NormPayload` objects."""

    raw_text = document.metadata.get("raw_text") or document.raw_text
    base_text = document.metadata.get("clean_text") or document.raw_text
    sections = _extract_sections(base_text)
    structure, current_division, current_chapter, current_section = _initialize_structure(base_text, sections)
    if current_chapter and not current_section:
        current_section = SectionNode(label=None, title=None, start=0, end=len(base_text))
        current_chapter.sections.append(current_section)
        structure.sections.append(current_section)
    if not current_chapter and current_division:
        current_chapter = ChapterNode(label=None, title=None, start=0, end=len(base_text))
        current_division.chapters.append(current_chapter)
        structure.chapters.append(current_chapter)
        if not current_section:
            current_section = SectionNode(label=None, title=None, start=0, end=len(base_text))
            current_chapter.sections.append(current_section)
            structure.sections.append(current_section)
    article_slices = _split_by_article(base_text)
    norms: List[NormPayload] = []
    for article_slice in article_slices:
        article_node = ArticleNode(
            label=article_slice.label,
            title=_extract_heading_title(article_slice.text),
            start=article_slice.start,
            end=article_slice.end,
        )
        structure.articles.append(article_node)
        if current_section:
            current_section.articles.append(article_node)
        elif current_chapter:
            temp_section = SectionNode(label=None, title=None, start=article_slice.start, end=article_slice.end, articles=[article_node])
            current_chapter.sections.append(temp_section)
            structure.sections.append(temp_section)
        part_slices = _split_by_part(article_slice)
        if not part_slices:
            part_slices = [TextSlice(label=None, start=article_slice.start, end=article_slice.end, text=article_slice.text)]
        for part_slice in part_slices:
            part_node = PartNode(label=part_slice.label, start=part_slice.start, end=part_slice.end, text=part_slice.text)
            article_node.parts.append(part_node)
            point_slices = _split_by_point(part_slice)
            if not point_slices:
                point_slices = [PointSlice(point=None, subpoint=None, start=part_slice.start, end=part_slice.end, text=part_slice.text)]
            for point_slice in point_slices:
                point_norms, point_node = _build_norms_for_slice(
                    document=document,
                    context=context,
                    sections=sections,
                    article_label=article_slice.label,
                    part_label=part_slice.label,
                    slice_=point_slice,
                    raw_text=raw_text,
                )
                norms.extend(point_norms)
                part_node.points.append(point_node)
    return ParsedDocument(norms=norms, structure=structure)


def _initialize_structure(
    text: str, sections: Dict[str, Optional[str]]
) -> tuple[DocumentStructure, Optional[DivisionNode], Optional[ChapterNode], Optional[SectionNode]]:
    structure = DocumentStructure()
    current_division: Optional[DivisionNode] = None
    current_chapter: Optional[ChapterNode] = None
    current_section: Optional[SectionNode] = None
    if sections["division"]:
        current_division = DivisionNode(label=sections["division"], title=None, start=0, end=len(text))
        structure.divisions.append(current_division)
    if sections["chapter"]:
        current_chapter = ChapterNode(label=sections["chapter"], title=None, start=0, end=len(text))
        structure.chapters.append(current_chapter)
        if current_division:
            current_division.chapters.append(current_chapter)
    if sections["section"]:
        current_section = SectionNode(label=sections["section"], title=None, start=0, end=len(text))
        structure.sections.append(current_section)
        if current_chapter:
            current_chapter.sections.append(current_section)
        elif current_division:
            shadow_chapter = ChapterNode(label=None, title=None, start=0, end=len(text), sections=[current_section])
            current_division.chapters.append(shadow_chapter)
    return structure, current_division, current_chapter, current_section


def _split_by_article(text: str) -> List[TextSlice]:
    matches = list(ARTICLE_RE.finditer(text))
    if not matches:
        return [TextSlice(label=None, start=0, end=len(text), text=text)]
    slices: List[TextSlice] = []
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        slices.append(TextSlice(label=match.group(1), start=start, end=end, text=text[start:end]))
    return slices


def _split_by_part(article: TextSlice) -> List[TextSlice]:
    text = article.text
    matches = list(PART_RE.finditer(text))
    if not matches:
        return []
    slices: List[TextSlice] = []
    current_start = 0
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        slices.append(
            TextSlice(
                label=match.group(1),
                start=article.start + start,
                end=article.start + end,
                text=text[start:end],
            )
        )
        current_start = end
    remainder = text[current_start:]
    if remainder.strip():
        slices.append(
            TextSlice(
                label=None,
                start=article.start + current_start,
                end=article.end,
                text=remainder,
            )
        )
    return slices


def _split_by_point(part: TextSlice) -> List[PointSlice]:
    text = part.text
    matches = list(POINT_RE.finditer(text))
    if not matches:
        return []
    slices: List[PointSlice] = []
    last_index = 0
    current_point: Optional[str] = None
    pending_preface = ""
    for idx, match in enumerate(matches):
        start = match.start()
        if start > last_index:
            preface = text[last_index:start]
            if preface.strip():
                if current_point is None:
                    pending_preface = preface
                else:
                    slices.append(
                        PointSlice(
                            point=current_point,
                            subpoint=None,
                            start=part.start + last_index,
                            end=part.start + start,
                            text=preface,
                        )
                    )
        label = match.group(2).strip()
        kind = match.group(1).lower()
        if kind.startswith("пункт"):
            current_point = label
            point_label = label
            subpoint_label = None
        else:
            point_label = current_point
            subpoint_label = label
        next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        segment_text = text[start:next_start]
        if pending_preface:
            segment_text = pending_preface + segment_text
            start -= len(pending_preface)
            pending_preface = ""
        slices.append(
            PointSlice(
                point=point_label,
                subpoint=subpoint_label,
                start=part.start + start,
                end=part.start + next_start,
                text=segment_text,
            )
        )
        last_index = next_start
    tail = text[last_index:]
    if tail.strip():
        slices.append(
            PointSlice(
                point=current_point,
                subpoint=None,
                start=part.start + last_index,
                end=part.end,
                text=tail,
            )
        )
    return slices


def _build_norms_for_slice(
    *,
    document: Document,
    context: ParsingContext,
    sections: Dict[str, Optional[str]],
    article_label: Optional[str],
    part_label: Optional[str],
    slice_: PointSlice,
    raw_text: str,
) -> Tuple[List[NormPayload], PointNode]:
    raw_segment = raw_text[slice_.start : slice_.end]
    stripped = _strip_headings(raw_segment)
    text_without_amendments, amendments = extract_amendments(stripped)
    text_without_annotations, annotations = extract_annotations(text_without_amendments)
    clean_text = _normalize_whitespace(text_without_annotations)
    cross_refs = extract_cross_refs(clean_text, context.law_code)
    point_node = PointNode(
        label=slice_.point,
        text=clean_text,
        start=slice_.start,
        end=slice_.end,
        subpoints=[],
    )
    base_payload = _create_payload(
        context=context,
        sections=sections,
        article_label=article_label or "",
        part_label=part_label,
        point_label=slice_.point,
        subpoint_label=slice_.subpoint,
        raw_text=raw_segment.strip(),
        clean_text=clean_text,
        amendments=amendments,
        annotations=annotations,
        cross_refs=cross_refs,
        span=(slice_.start, slice_.end),
        page_spans=document.metadata.get("page_spans", []),
        file_origin=str(document.path),
    )
    payloads = [base_payload]
    if slice_.subpoint is None:
        list_segments = split_semicolon_list(stripped, slice_.start)
        for idx, segment in enumerate(list_segments, start=1):
            segment_clean = _normalize_whitespace(segment.text)
            subpoint_node = SubpointNode(label=str(idx), text=segment_clean, start=segment.offsets[0], end=segment.offsets[1])
            point_node.subpoints.append(subpoint_node)
            payloads.append(
                _create_payload(
                    context=context,
                    sections=sections,
                    article_label=article_label or "",
                    part_label=part_label,
                    point_label=slice_.point,
                    subpoint_label=str(idx),
                    raw_text=segment.text,
                    clean_text=segment_clean,
                    amendments=[],
                    annotations=[],
                    cross_refs=extract_cross_refs(segment_clean, context.law_code),
                    span=segment.offsets,
                    page_spans=document.metadata.get("page_spans", []),
                    file_origin=str(document.path),
                )
            )
    return payloads, point_node


def _create_payload(
    *,
    context: ParsingContext,
    sections: Dict[str, Optional[str]],
    article_label: str,
    part_label: Optional[str],
    point_label: Optional[str],
    subpoint_label: Optional[str],
    raw_text: str,
    clean_text: str,
    amendments: List[Dict[str, str]],
    annotations: List[Dict[str, str]],
    cross_refs: List[Dict[str, Optional[str]]],
    span: Tuple[int, int],
    page_spans: Iterable[Sequence[int]],
    file_origin: Optional[str],
) -> NormPayload:
    citation_parts: List[str] = [context.law_code]
    if sections.get("division"):
        citation_parts.append(sections["division"])
    if sections.get("chapter"):
        citation_parts.append(sections["chapter"])
    if sections.get("section"):
        citation_parts.append(sections["section"])
    if article_label:
        citation_parts.append(f"ст. {article_label}")
    if part_label:
        citation_parts.append(f"ч. {part_label}")
    if point_label:
        citation_parts.append(f"п. {point_label}")
    if subpoint_label:
        citation_parts.append(f"подп. {subpoint_label}")
    citation = ", ".join(filter(None, citation_parts)) + f" (ред. {context.version_id})"
    tokens_est = _estimate_tokens(clean_text)
    page_pairs = [tuple(map(int, pair)) for pair in page_spans]
    return NormPayload.build(
        jurisdiction=context.jurisdiction,
        act_type=context.act_type,
        division=sections.get("division"),
        chapter=sections.get("chapter"),
        section=sections.get("section"),
        law_code=context.law_code,
        law_full_title=context.law_full_title,
        article=article_label or "",
        part=part_label,
        point=point_label,
        subpoint=subpoint_label,
        version_id=context.version_id,
        valid_from=context.valid_from,
        valid_to=context.valid_to,
        supersedes=context.supersedes,
        source_url=context.source_url,
        citation=citation,
        lang=context.lang,
        raw_text=raw_text.strip(),
        clean_text=clean_text.strip(),
        amendments=amendments,
        annotations=annotations,
        cross_refs=cross_refs,
        page_spans=page_pairs,
        span_offsets=[(int(span[0]), int(span[1]))],
        tokens_est=tokens_est,
        file_origin=file_origin,
    )


def _extract_sections(text: str) -> Dict[str, Optional[str]]:
    sections: Dict[str, Optional[str]] = {"division": None, "chapter": None, "section": None}
    if division := DIVISION_RE.search(text):
        sections["division"] = f"Подраздел {division.group('num')}"
    if chapter := CHAPTER_RE.search(text):
        sections["chapter"] = f"Глава {chapter.group('num')}"
    if section := SECTION_RE.search(text):
        sections["section"] = f"Раздел {section.group('num')}"
    return sections


def _extract_heading_title(text: str) -> Optional[str]:
    lines = text.strip().splitlines()
    if len(lines) >= 2:
        first = lines[0].strip()
        second = lines[1].strip()
        if not second.lower().startswith("ч") and not second.lower().startswith("п"):
            return second
    return None


def _strip_headings(text: str) -> str:
    result = text.strip()
    for pattern in HEADER_STRIP_PATTERNS:
        result = pattern.sub("", result, count=1).lstrip()
    return result.strip()


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _estimate_tokens(text: str) -> int:
    words = text.split()
    if not words:
        return 0
    return max(1, len(words) * 4 // 3)


__all__ = ["ParsedDocument", "ParsingContext", "parse_document"]
