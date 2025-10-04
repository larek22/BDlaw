"""Parse Russian legal texts into structured norm payloads."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from bdlaw.index.payload import NormPayload
from bdlaw.ingest.base import Document

ARTICLE_RE = re.compile(r"^\s*Ст(?:атья|\.)\s+(\d+(?:\.\d+)?)\b", re.IGNORECASE | re.MULTILINE)
PART_RE = re.compile(r"^\s*Ч(?:асть|\.)\s+([IVXLC\d]+)\b", re.IGNORECASE | re.MULTILINE)
POINT_RE = re.compile(r"(?<!\S)(Пункт|Подпункт)\s+([\dа-яА-Яivxlc\)\(\.]+)\b", re.IGNORECASE)
DIVISION_RE = re.compile(r"^Подраздел\s+(?P<num>\d+)\.?\s*(?P<title>.*)$", re.MULTILINE)
CHAPTER_RE = re.compile(r"^Глава\s+(?P<num>\d+)\.?\s*(?P<title>.*)$", re.MULTILINE)
SECTION_RE = re.compile(r"^Раздел\s+(?P<num>\d+)\.?\s*(?P<title>.*)$", re.MULTILINE)
AMENDMENT_RE = re.compile(
    r"\(в\s+ред\.\s+Федерального\s+закона\s+от\s+(\d{2}\.\d{2}\.\d{4})\s+№\s*([\d\-А-Яа-яA-Za-z]+)\)",
    re.IGNORECASE,
)
CC_RULING_RE = re.compile(
    r"\((?:постановление|определение)\s+Конституционного\s+суда\s+РФ\s+от\s+(\d{2}\.\d{2}\.\d{4})\s+№\s*([\d\-А-Яа-яA-Za-z]+)[^\)]*\)",
    re.IGNORECASE,
)
CROSS_REF_RE = re.compile(
    r"ст(?:атья|\.)?\s+(?P<article>\d+(?:\.\d+)?)(?:\s*,?\s*ч(?:асть|\.)?\s*(?P<part>[IVXLC\d]+))?(?:\s+(?P<law>[А-ЯЁA-Z]{2,}(?:\s+[А-ЯЁA-Z]{2,}){0,3}))?",
    re.IGNORECASE,
)
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

    base_text = document.metadata.get("clean_text") or document.raw_text
    raw_text = document.metadata.get("raw_text") or document.raw_text
    sections = _extract_sections(base_text)
    article_slices = _split_by_article(base_text)
    norms: List[NormPayload] = []
    for article in article_slices:
        part_slices = _split_by_part(article)
        for part in part_slices:
            point_slices = _split_by_point(part)
            for slice_ in point_slices:
                norms.extend(
                    _build_norm_payloads(
                        document=document,
                        context=context,
                        sections=sections,
                        article_label=article.label,
                        part_label=part.label,
                        slice_=slice_,
                        raw_text=raw_text,
                    )
                )
    return ParsedDocument(norms=norms)


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
        return [TextSlice(label=None, start=article.start, end=article.end, text=text)]
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
    if current_start < len(text):
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
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        label = match.group(2).strip()
        kind = match.group(1).lower()
        if kind.startswith("пункт"):
            current_point = label
            point_label = label
            subpoint_label = None
        else:
            point_label = current_point
            subpoint_label = label
        segment_text = text[start:end]
        if pending_preface:
            segment_text = pending_preface + segment_text
            start = start - len(pending_preface)
            pending_preface = ""
        slices.append(
            PointSlice(
                point=point_label,
                subpoint=subpoint_label,
                start=part.start + start,
                end=part.start + end,
                text=segment_text,
            )
        )
        last_index = end
    if not matches:
        slices.append(PointSlice(point=None, subpoint=None, start=part.start, end=part.end, text=text))
    elif last_index < len(text):
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


def _extract_sections(text: str) -> Dict[str, Optional[str]]:
    sections: Dict[str, Optional[str]] = {"division": None, "chapter": None, "section": None}
    if division := DIVISION_RE.search(text):
        sections["division"] = f"Подраздел {division.group('num')}"
    if chapter := CHAPTER_RE.search(text):
        sections["chapter"] = f"Глава {chapter.group('num')}"
    if section := SECTION_RE.search(text):
        sections["section"] = f"Раздел {section.group('num')}"
    return sections


def _strip_headings(text: str) -> str:
    result = text.strip()
    for pattern in HEADER_STRIP_PATTERNS:
        result = pattern.sub("", result, count=1).lstrip()
    return result.strip()


def _extract_amendments(text: str) -> Tuple[List[Dict[str, str]], str]:
    amendments: List[Dict[str, str]] = []

    def repl(match: re.Match[str]) -> str:
        date_str = match.group(1)
        law_no = match.group(2)
        amendments.append({"date": _isoformat(date_str), "law_no": law_no, "scope": "article"})
        return ""

    cleaned = AMENDMENT_RE.sub(repl, text)
    return amendments, cleaned


def _extract_annotations(text: str) -> Tuple[List[Dict[str, str]], str]:
    annotations: List[Dict[str, str]] = []

    def repl(match: re.Match[str]) -> str:
        date_str = match.group(1)
        number = match.group(2)
        content = match.group(0)
        annotations.append(
            {
                "type": "cc_ruling",
                "date": _isoformat(date_str),
                "no": number,
                "text": content,
            }
        )
        return ""

    cleaned = CC_RULING_RE.sub(repl, text)
    return annotations, cleaned


def _extract_cross_refs(text: str, default_law: str) -> List[Dict[str, Optional[str]]]:
    refs: Dict[Tuple[str, Optional[str]], Dict[str, Optional[str]]] = {}
    for match in CROSS_REF_RE.finditer(text):
        article = match.group("article")
        part = match.group("part")
        law = match.group("law")
        key = (article, part)
        refs[key] = {
            "law_code": law.strip() if law else default_law,
            "article": article,
            "part": part,
        }
    return list(refs.values())


def _split_semicolon_list(text: str, start_offset: int) -> List[Tuple[str, Tuple[int, int]]]:
    segments: List[Tuple[str, Tuple[int, int]]] = []
    depth = 0
    in_quote = False
    quote_char: Optional[str] = None
    segment_start = 0
    for idx, char in enumerate(text):
        if char in '"\'«»':
            if in_quote and char == quote_char:
                in_quote = False
                quote_char = None
            elif not in_quote:
                in_quote = True
                quote_char = char
        elif char == "(":
            depth += 1
        elif char == ")" and depth > 0:
            depth -= 1
        elif char == ";" and depth == 0 and not in_quote:
            remainder = text[idx + 1 :]
            remainder_stripped = remainder.lstrip()
            if remainder_stripped and (remainder_stripped[0].islower() or remainder_stripped[0].isdigit()):
                segment_text = text[segment_start:idx].strip(" ;\n")
                if segment_text:
                    segments.append((segment_text, (start_offset + segment_start, start_offset + idx)))
                segment_start = idx + 1
    tail = text[segment_start:].strip()
    if tail:
        segments.append((tail.strip(" ;\n"), (start_offset + segment_start, start_offset + len(text))))
    if len(segments) <= 1:
        return []
    return segments


def _build_norm_payloads(
    *,
    document: Document,
    context: ParsingContext,
    sections: Dict[str, Optional[str]],
    article_label: Optional[str],
    part_label: Optional[str],
    slice_: PointSlice,
    raw_text: str,
) -> List[NormPayload]:
    raw_segment = raw_text[slice_.start : slice_.end]
    stripped = _strip_headings(raw_segment)
    amendments, without_amendments = _extract_amendments(stripped)
    annotations, without_annotations = _extract_annotations(without_amendments)
    clean_text = _normalize_whitespace(without_annotations)
    cross_refs = _extract_cross_refs(clean_text, context.law_code)
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
        list_segments = _split_semicolon_list(stripped, slice_.start)
        for idx, (segment_text, (seg_start, seg_end)) in enumerate(list_segments, start=1):
            segment_clean = _normalize_whitespace(segment_text)
            payloads.append(
                _create_payload(
                    context=context,
                    sections=sections,
                    article_label=article_label or "",
                    part_label=part_label,
                    point_label=slice_.point,
                    subpoint_label=str(idx),
                    raw_text=segment_text,
                    clean_text=_normalize_whitespace(segment_text),
                    amendments=[],
                    annotations=[],
                    cross_refs=_extract_cross_refs(segment_clean, context.law_code),
                    span=(seg_start, seg_end),
                    page_spans=document.metadata.get("page_spans", []),
                    file_origin=str(document.path),
                )
            )
    return payloads


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
        article=article_label or "",  # required
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


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _estimate_tokens(text: str) -> int:
    words = text.split()
    if not words:
        return 0
    return max(1, len(words) * 4 // 3)


def _isoformat(date_str: str) -> str:
    day, month, year = date_str.split(".")
    return f"{year}-{month}-{day}"


__all__ = [
    "ParsedDocument",
    "ParsingContext",
    "parse_document",
]
