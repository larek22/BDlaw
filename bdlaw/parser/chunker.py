"""Chunking utilities for norms."""

from __future__ import annotations

import dataclasses
from typing import Iterable, List

from bdlaw.parser.structure import Norm
from bdlaw.settings.config import ChunkingConfig


def chunk_norm(norm: Norm, config: ChunkingConfig) -> List[Norm]:
    words = norm.clean_text.split()
    if len(words) <= config.max_words:
        return [norm]
    chunks: List[Norm] = []
    step = config.max_words - config.overlap_words
    for start in range(0, len(words), step):
        end = min(len(words), start + config.max_words)
        window = words[start:end]
        chunk_text = " ".join(window)
        clone = dataclasses.replace(norm)
        clone.clean_text = chunk_text
        clone.raw_text = chunk_text
        clone.span_offsets = [[start, end]]
        chunks.append(clone)
        if end == len(words):
            break
    return chunks


def chunk_norms(norms: Iterable[Norm], config: ChunkingConfig) -> List[Norm]:
    result: List[Norm] = []
    for norm in norms:
        result.extend(chunk_norm(norm, config))
    return result


__all__ = ["chunk_norm", "chunk_norms"]
