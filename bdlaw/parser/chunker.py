"""Chunking utilities for norms."""

from __future__ import annotations

import re
from typing import Iterable, List

from bdlaw.index.schema import NormPayload
from bdlaw.normalize.offsets import build_offset_map, project_relative_span
from bdlaw.settings.config import ChunkingConfig


def chunk_norm(norm: NormPayload, config: ChunkingConfig) -> List[NormPayload]:
    word_matches = list(re.finditer(r"\S+", norm.clean_text))
    if len(word_matches) <= config.max_words:
        return [norm]
    chunks: List[NormPayload] = []
    step = max(1, config.max_words - config.overlap_words)
    mapping = build_offset_map(norm.raw_text, norm.clean_text)
    base_span = norm.span_offsets[0] if norm.span_offsets else (0, len(norm.raw_text))
    for start in range(0, len(word_matches), step):
        end = min(len(word_matches), start + config.max_words)
        start_char = word_matches[start].start()
        end_char = word_matches[end - 1].end()
        clean_slice = norm.clean_text[start_char:end_char]
        raw_start_rel, raw_end_rel = project_relative_span((start_char, end_char), mapping, len(norm.raw_text))
        raw_slice = norm.raw_text[raw_start_rel:raw_end_rel]
        chunk = NormPayload.build(
            jurisdiction=norm.jurisdiction,
            act_type=norm.act_type,
            division=norm.division,
            chapter=norm.chapter,
            section=norm.section,
            law_code=norm.law_code,
            law_full_title=norm.law_full_title,
            article=norm.article,
            part=norm.part,
            point=norm.point,
            subpoint=norm.subpoint,
            version_id=norm.version_id,
            valid_from=norm.valid_from,
            valid_to=norm.valid_to,
            supersedes=norm.supersedes,
            source_url=str(norm.source_url) if norm.source_url else None,
            citation=norm.citation,
            lang=norm.lang,
            raw_text=raw_slice,
            clean_text=clean_slice,
            amendments=list(norm.amendments),
            annotations=list(norm.annotations),
            cross_refs=list(norm.cross_refs),
            page_spans=list(norm.page_spans),
            span_offsets=[(base_span[0] + raw_start_rel, base_span[0] + raw_end_rel)],
            tokens_est=max(1, (end - start) * 4 // 3),
            file_origin=norm.file_origin,
            ingested_at=norm.ingested_at,
        )
        chunks.append(chunk)
        if end == len(word_matches):
            break
    return chunks


def chunk_norms(norms: Iterable[NormPayload], config: ChunkingConfig) -> List[NormPayload]:
    result: List[NormPayload] = []
    for norm in norms:
        result.extend(chunk_norm(norm, config))
    return result


__all__ = ["chunk_norm", "chunk_norms"]
