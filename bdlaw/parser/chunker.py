"""Chunking utilities for norms."""

from __future__ import annotations

from typing import Iterable, List

from bdlaw.index.payload import NormPayload
from bdlaw.settings.config import ChunkingConfig


def chunk_norm(norm: NormPayload, config: ChunkingConfig) -> List[NormPayload]:
    words = norm.clean_text.split()
    if len(words) <= config.max_words:
        return [norm]
    chunks: List[NormPayload] = []
    step = config.max_words - config.overlap_words
    for start in range(0, len(words), step):
        end = min(len(words), start + config.max_words)
        window = words[start:end]
        chunk_text = " ".join(window)
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
            raw_text=chunk_text,
            clean_text=chunk_text,
            amendments=list(norm.amendments),
            annotations=list(norm.annotations),
            cross_refs=list(norm.cross_refs),
            page_spans=list(norm.page_spans),
            span_offsets=[(start, end)],
            tokens_est=max(1, len(window) * 4 // 3),
            file_origin=norm.file_origin,
            ingested_at=norm.ingested_at,
        )
        chunks.append(chunk)
        if end == len(words):
            break
    return chunks


def chunk_norms(norms: Iterable[NormPayload], config: ChunkingConfig) -> List[NormPayload]:
    result: List[NormPayload] = []
    for norm in norms:
        result.extend(chunk_norm(norm, config))
    return result


__all__ = ["chunk_norm", "chunk_norms"]
