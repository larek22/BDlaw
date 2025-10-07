from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import List

from .readers.base import DocumentText


_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def _split_sentences(text: str) -> List[str]:
    sentences: List[str] = []
    start = 0
    for match in _SENTENCE_END_RE.finditer(text):
        end = match.end()
        sentence = text[start:end].strip()
        if sentence:
            sentences.append(sentence)
        start = match.end()
    tail = text[start:].strip()
    if tail:
        sentences.append(tail)
    if not sentences:
        sentences = [text]
    return sentences


@dataclass
class Chunk:
    doc_id: str
    path: str
    page: int | None
    chunk_index: int
    offset: int
    text: str
    sha: str


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_document(
    document: DocumentText,
    chunk_size_chars: int = 1200,
    chunk_overlap_chars: int = 120,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    chunk_index = 0
    for page_number, page_text in enumerate(document.pages, start=1):
        sentences = _split_sentences(page_text)
        buffer = ""
        offset = 0
        for sentence in sentences:
            sentence_text = sentence.strip()
            if not sentence_text:
                continue
            if buffer and len(buffer) + len(sentence_text) > chunk_size_chars:
                chunk_text = buffer.strip()
                if chunk_text:
                    sha = _hash_text(chunk_text)
                    chunks.append(
                        Chunk(
                            doc_id=document.doc_id,
                            path=str(document.path),
                            page=page_number,
                            chunk_index=chunk_index,
                            offset=offset,
                            text=chunk_text,
                            sha=sha,
                        )
                    )
                    chunk_index += 1
                    offset += len(chunk_text)
                    buffer = chunk_text[-chunk_overlap_chars:]
            if buffer:
                buffer += " " + sentence_text
            else:
                buffer = sentence_text
        if buffer:
            chunk_text = buffer.strip()
            if chunk_text:
                sha = _hash_text(chunk_text)
                chunks.append(
                    Chunk(
                        doc_id=document.doc_id,
                        path=str(document.path),
                        page=page_number,
                        chunk_index=chunk_index,
                        offset=offset,
                        text=chunk_text,
                        sha=sha,
                    )
                )
                chunk_index += 1
            buffer = ""
    return chunks


__all__ = ["Chunk", "chunk_document"]
