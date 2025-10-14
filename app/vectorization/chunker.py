"""Deterministic chunker that applies a :class:`VectorizationPlan`."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, TYPE_CHECKING

from .models import VectorizationPlan
from .normalizer import Block
from .tokenization import DEFAULT_TOKENIZER, Tokenizer

if TYPE_CHECKING:
    from .models import RecordChunking


@dataclass(slots=True)
class Chunk:
    text: str
    tokens: int
    start: int
    end: int
    meta: Dict[str, object]
    chunk_id: str


def _make_chunk_id(plan: VectorizationPlan, *, start: int, end: int, index: int, source: str) -> str:
    pattern = plan.id_strategy.pattern
    candidate = pattern.replace("{start}", str(start)).replace("{end}", str(end)).replace(
        "{index}", str(index)
    )
    if "{uuid" in pattern:
        candidate = candidate.replace("{uuid}", uuid.uuid4().hex)
    if plan.id_strategy.ensure_uuid_if_missing and "{" not in pattern:
        candidate = f"{candidate}:{uuid.uuid4().hex}"
    return candidate


@dataclass(slots=True)
class ChunkerResult:
    chunks: List[Chunk]
    token_usage: int


def apply_plan(
    blocks: Sequence[Block],
    plan: VectorizationPlan,
    *,
    tokenizer: Tokenizer | None = None,
) -> ChunkerResult:
    tokenizer = tokenizer or DEFAULT_TOKENIZER
    policy = plan.chunking_policy
    chunks: List[Chunk] = []
    overlap_tokens = policy.overlap_tokens
    window_texts: List[str] = []
    window_tokens: List[int] = []
    window_token_total = 0
    chunk_index = 0

    def flush_chunk(final: bool = False) -> None:
        nonlocal window_texts, window_tokens, window_token_total, chunk_index, window_meta
        if not window_texts:
            return
        text = "\n".join(window_texts).strip()
        if not text and plan.quality_checks.forbid_empty_text:
            window_texts = []
            window_tokens = []
            window_token_total = 0
            return
        start_pos = 0
        end_pos = 0
        if window_meta:
            start_pos = min(meta.get("start", 0) for meta in window_meta if isinstance(meta, dict))
            end_pos = max(meta.get("end", 0) for meta in window_meta if isinstance(meta, dict))
        chunk_id = _make_chunk_id(
            plan,
            start=start_pos,
            end=end_pos,
            index=chunk_index,
            source=blocks[0].path if blocks else "",
        )
        chunk_meta = {
            "source_file": blocks[0].path if blocks else "",
            "index": chunk_index,
        }
        chunks.append(
            Chunk(
                text=text,
                tokens=sum(window_tokens),
                start=start_pos,
                end=end_pos,
                meta=chunk_meta,
                chunk_id=chunk_id,
            )
        )
        chunk_index += 1
        if final:
            window_texts = []
            window_tokens = []
            window_meta = []
            window_token_total = 0
            return
        if overlap_tokens <= 0:
            window_texts = []
            window_tokens = []
            window_meta = []
            window_token_total = 0
            return
        new_texts: List[str] = []
        new_tokens: List[int] = []
        new_meta: List[Dict[str, int]] = []
        total = 0
        for text_piece, token_count, meta in reversed(list(zip(window_texts, window_tokens, window_meta))):
            new_texts.insert(0, text_piece)
            new_tokens.insert(0, token_count)
            new_meta.insert(0, meta)
            total += token_count
            if total >= overlap_tokens:
                break
        window_texts = new_texts
        window_tokens = new_tokens
        window_meta = new_meta
        window_token_total = sum(new_tokens)

    window_meta: List[Dict[str, int]] = []
    token_usage = 0

    def add_block_text(block: Block, text: str) -> None:
        nonlocal window_token_total, token_usage
        if not text:
            return
        tokens = tokenizer.count(text)
        if tokens == 0 and plan.quality_checks.forbid_empty_text:
            return
        token_usage += tokens
        if tokens > policy.max_tokens:
            slices = _split_long_text(text, tokenizer, policy.max_tokens)
            for slice_text in slices:
                add_block_text(block, slice_text)
            return
        if window_token_total + tokens > policy.max_tokens and window_texts:
            flush_chunk()
        window_texts.append(text)
        window_tokens.append(tokens)
        window_meta.append(block.position or {"start": 0, "end": 0})
        window_token_total += tokens

    if policy.mode == "by_records":
        record_policy = policy.record_chunking
        buffer: List[Block] = []
        for block in blocks:
            if block.type != "record":
                continue
            buffer.append(block)
            if len(buffer) >= record_policy.records_per_chunk:
                record_texts = [_record_payload_text(b, record_policy) for b in buffer]
                combined = "\n".join(record_texts)
                add_block_text(buffer[-1], combined)
                buffer = []
        if buffer:
            record_texts = [_record_payload_text(b, record_policy) for b in buffer]
            combined = "\n".join(record_texts)
            add_block_text(buffer[-1], combined)
        flush_chunk(final=True)
        return ChunkerResult(chunks=chunks, token_usage=token_usage)

    current_heading: Dict[int, str] = {}
    split_levels = {int(level[1:]) if level.startswith("h") and level[1:].isdigit() else 0 for level in policy.split_on_headings}

    for block in blocks:
        if block.type == "heading":
            level = int(block.attrs.get("level", 1))
            current_heading[level] = block.text
            if split_levels and level in split_levels and window_texts:
                flush_chunk()
            heading_text = block.text.strip()
            if heading_text:
                add_block_text(block, heading_text)
            continue
        if block.type == "list_item" and policy.keep_lists_intact:
            add_block_text(block, f"- {block.text.strip()}")
            continue
        if block.type == "table" and policy.keep_tables_intact:
            add_block_text(block, block.text.strip())
            continue
        if block.type == "paragraph":
            text = block.text.strip()
            if policy.merge_short_paragraphs_under_tokens > 0:
                tokens = tokenizer.count(text)
                if (
                    tokens < policy.merge_short_paragraphs_under_tokens
                    and window_token_total + tokens <= policy.max_tokens
                ):
                    add_block_text(block, text)
                    continue
            add_block_text(block, text)
            continue
        add_block_text(block, block.text.strip())

    flush_chunk(final=True)
    return ChunkerResult(chunks=chunks, token_usage=token_usage)


def _record_payload_text(block: Block, policy: "RecordChunking") -> str:
    data = block.attrs.get("data")
    if isinstance(data, dict):
        filtered = {}
        include = policy.fields_include or ["*"]
        exclude = set(policy.fields_exclude or [])
        if "*" in include:
            for key, value in data.items():
                if key in exclude:
                    continue
                filtered[key] = value
        else:
            for key in include:
                if key in exclude:
                    continue
                filtered[key] = data.get(key)
        return json.dumps(filtered, ensure_ascii=False)
    return block.text


def _split_long_text(text: str, tokenizer: Tokenizer, max_tokens: int) -> List[str]:
    tokens = tokenizer.encode(text)
    if not tokens:
        return [text]
    slices: List[str] = []
    for start in range(0, len(tokens), max_tokens):
        end = min(start + max_tokens, len(tokens))
        window = tokens[start:end]
        slices.append(tokenizer.decode(window))
    return [slice_text for slice_text in slices if slice_text] or [text]


__all__ = ["Chunk", "ChunkerResult", "apply_plan"]
