"""Deterministic chunker that applies a :class:`VectorizationPlan`."""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, TYPE_CHECKING

from .models import ChunkingPolicy, VectorizationPlan
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


def _make_chunk_id(
    plan: VectorizationPlan,
    *,
    start: int,
    end: int,
    index: int,
    source: str,
) -> str:
    pattern = plan.id_strategy.pattern
    source_basename = Path(source).stem if source else "document"
    candidate = (
        pattern.replace("{start}", str(start))
        .replace("{end}", str(end))
        .replace("{index}", str(index))
        .replace("{source_basename}", source_basename)
        .replace("{source}", source)
    )
    seed = f"{source}|{start}|{end}|{index}"
    if "{uuid" in pattern:
        candidate = candidate.replace(
            "{uuid}", uuid.uuid5(uuid.NAMESPACE_URL, seed).hex
        )
    if plan.id_strategy.ensure_uuid_if_missing and "{" not in pattern:
        candidate = f"{candidate}:{uuid.uuid5(uuid.NAMESPACE_URL, seed).hex}"
    return candidate


@dataclass(slots=True)
class ChunkerResult:
    chunks: List[Chunk]
    token_usage: int


def _split_long_text(text: str, policy: ChunkingPolicy, tokenizer: Tokenizer) -> List[str]:
    """Split text into deterministic slices within the token cap."""
    if not text:
        return []

    max_tokens = max(1, policy.max_tokens)
    overlap = max(0, min(policy.overlap_tokens, max_tokens - 1))
    tokens = tokenizer.encode(text)
    if not tokens:
        return [text]

    step = max(1, max_tokens - overlap)
    slices: List[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        window = tokens[start:end]
        if not window:
            break
        decoded = tokenizer.decode(window)
        counted = tokenizer.encode(decoded)
        while len(counted) > max_tokens and len(window) > 1:
            window = window[:-1]
            decoded = tokenizer.decode(window)
            counted = tokenizer.encode(decoded)
        slices.append(decoded)
        if end >= len(tokens):
            break
        start += step

    return [segment for segment in slices if segment] or [text]


def apply_plan(
    blocks: Sequence[Block],
    plan: VectorizationPlan,
    *,
    tokenizer: Tokenizer | None = None,
) -> ChunkerResult:
    tokenizer = tokenizer or DEFAULT_TOKENIZER
    policy = plan.chunking_policy
    overlap_tokens = max(0, min(policy.overlap_tokens, policy.max_tokens))

    chunks: List[Chunk] = []
    window_texts: List[str] = []
    window_tokens: List[int] = []
    window_meta: List[Dict[str, object]] = []
    window_token_total = 0

    token_usage = 0
    chunk_index = 0
    last_start = -1
    last_end = -1

    named_hierarchy: Dict[str, str] = {}
    current_heading: Dict[int, str] = {}
    heading_regex = plan.hierarchy_rules.heading_regex or {}

    def _reset_window() -> None:
        nonlocal window_texts, window_tokens, window_meta, window_token_total
        window_texts = []
        window_tokens = []
        window_meta = []
        window_token_total = 0

    def _append_overlap(chunk: Chunk, meta_list: Sequence[Dict[str, object]]) -> None:
        nonlocal window_texts, window_tokens, window_meta, window_token_total
        if overlap_tokens <= 0:
            return
        encoded = tokenizer.encode(chunk.text)
        if not encoded:
            return
        tail_len = min(overlap_tokens, len(encoded))
        if tail_len <= 0:
            return
        overlap_slice = encoded[-tail_len:]
        window_texts = [tokenizer.decode(overlap_slice)]
        window_tokens = [len(overlap_slice)]
        window_meta = list(meta_list)
        window_token_total = sum(window_tokens)

    def _emit_chunk(text: str, meta_list: Sequence[Dict[str, object]]) -> Chunk | None:
        nonlocal chunk_index, last_start, last_end

        cleaned_text = text.strip("\n")
        if not cleaned_text and plan.quality_checks.forbid_empty_text:
            return None

        token_count = tokenizer.count(cleaned_text)
        if token_count == 0 and plan.quality_checks.forbid_empty_text:
            return None

        if token_count > policy.max_tokens:
            emitted: Chunk | None = None
            for slice_text in _split_long_text(cleaned_text, policy, tokenizer):
                emitted = _emit_chunk(slice_text, meta_list)
            return emitted

        start_candidates = [
            meta.get("start")
            for meta in meta_list
            if isinstance(meta, dict) and isinstance(meta.get("start"), int)
        ]
        end_candidates = [
            meta.get("end")
            for meta in meta_list
            if isinstance(meta, dict) and isinstance(meta.get("end"), int)
        ]

        if start_candidates:
            start_pos = min(start_candidates)
        else:
            start_pos = last_end + 1 if last_end >= 0 else 0
        if end_candidates:
            end_pos = max(end_candidates)
        else:
            end_pos = start_pos + max(len(cleaned_text), 1)

        if start_pos <= last_start:
            start_pos = last_start + 1
        if end_pos <= start_pos:
            end_pos = start_pos + max(len(cleaned_text), 1)
        if end_pos <= last_end:
            end_pos = last_end + max(len(cleaned_text), 1)

        source_path = blocks[0].path if blocks else ""
        chunk_id = _make_chunk_id(
            plan,
            start=start_pos,
            end=end_pos,
            index=chunk_index,
            source=source_path,
        )

        chunk_meta: Dict[str, object] = {
            "source_file": source_path,
            "index": chunk_index,
        }
        if named_hierarchy:
            chunk_meta["hierarchy"] = dict(named_hierarchy)
            for field in plan.hierarchy_rules.path_fields:
                if field in named_hierarchy:
                    chunk_meta[field] = named_hierarchy[field]

        heading_values = [value for _, value in sorted(current_heading.items())]
        title_candidate = (
            named_hierarchy.get("article")
            or named_hierarchy.get("chapter")
            or named_hierarchy.get("section")
            or (heading_values[-1] if heading_values else "")
        )
        if title_candidate:
            chunk_meta["title_text"] = title_candidate

        row_indices = [
            meta.get("attrs", {}).get("row_index")
            for meta in meta_list
            if isinstance(meta, dict)
            and isinstance(meta.get("attrs"), dict)
            and isinstance(meta.get("attrs", {}).get("row_index"), int)
        ]
        if row_indices:
            chunk_meta["row_index"] = row_indices[-1]

        if source_path and "doc_id" not in chunk_meta:
            chunk_meta["doc_id"] = Path(source_path).stem

        chunk = Chunk(
            text=cleaned_text,
            tokens=token_count,
            start=start_pos,
            end=end_pos,
            meta=chunk_meta,
            chunk_id=chunk_id,
        )
        chunk_meta["plan_chunk_id"] = chunk_id
        chunks.append(chunk)
        chunk_index += 1
        last_start = start_pos
        last_end = end_pos
        return chunk

    def flush_chunk(final: bool = False) -> None:
        if not window_texts:
            return

        previous_texts = list(window_texts)
        previous_meta = list(window_meta)
        total_tokens = sum(window_tokens)

        _reset_window()

        combined_text = "\n".join(previous_texts)
        if not combined_text and plan.quality_checks.forbid_empty_text:
            return

        if total_tokens > policy.max_tokens:
            segments = _split_long_text(combined_text, policy, tokenizer)
        else:
            segments = [combined_text]

        last_chunk_emitted: Chunk | None = None
        for segment in segments:
            emitted = _emit_chunk(segment, previous_meta)
            if emitted is not None:
                last_chunk_emitted = emitted

        if not final and last_chunk_emitted is not None:
            _append_overlap(last_chunk_emitted, previous_meta)

    def add_block_text(block: Block, text: str) -> None:
        nonlocal window_token_total, token_usage
        if not text:
            return

        tokens = tokenizer.count(text)
        if tokens == 0 and plan.quality_checks.forbid_empty_text:
            return

        if tokens > policy.max_tokens:
            for slice_text in _split_long_text(text, policy, tokenizer):
                add_block_text(block, slice_text)
            return

        token_usage += tokens

        if window_texts and window_token_total + tokens > policy.max_tokens:
            flush_chunk()

        window_texts.append(text)
        window_tokens.append(tokens)
        block_meta = {
            "start": (block.position or {}).get("start", 0),
            "end": (block.position or {}).get("end", 0),
            "type": block.type,
            "attrs": dict(block.attrs or {}),
            "text": block.text,
            "path": block.path,
        }
        window_meta.append(block_meta)
        window_token_total += tokens

    if policy.mode == "by_records":
        record_policy = policy.record_chunking
        buffer: List[Block] = []
        for block in blocks:
            if block.type != "record":
                continue
            buffer.append(block)
            if len(buffer) >= record_policy.records_per_chunk:
                record_text = "\n".join(
                    _record_payload_text(item, record_policy) for item in buffer
                )
                add_block_text(buffer[-1], record_text)
                flush_chunk()
                buffer = []
        if buffer:
            record_text = "\n".join(
                _record_payload_text(item, record_policy) for item in buffer
            )
            add_block_text(buffer[-1], record_text)
            flush_chunk()
        flush_chunk(final=True)
        return ChunkerResult(chunks=chunks, token_usage=token_usage)

    split_levels = {
        int(level[1:])
        for level in policy.split_on_headings
        if level.startswith("h") and level[1:].isdigit()
    }
    split_names = {
        name
        for name in policy.split_on_headings
        if not (name.startswith("h") and name[1:].isdigit())
    }

    for block in blocks:
        if block.type == "heading":
            level = int(block.attrs.get("level", 1))
            current_heading[level] = block.text
            for deeper in [lvl for lvl in list(current_heading) if lvl > level]:
                current_heading.pop(deeper, None)

            heading_text = block.text.strip()
            should_split = level in split_levels
            if split_names and heading_text:
                for name in split_names:
                    pattern = heading_regex.get(name)
                    if pattern and re.search(pattern, heading_text):
                        should_split = True
                        break
            if heading_text:
                for name, pattern in heading_regex.items():
                    if pattern and re.search(pattern, heading_text):
                        named_hierarchy[name] = heading_text
            if should_split and window_texts:
                flush_chunk()
            if heading_text:
                add_block_text(block, heading_text)
                flush_chunk()
            continue

        if block.type == "list_item" and policy.keep_lists_intact:
            add_block_text(block, f"- {block.text.strip()}")
            continue

        if block.type == "table" and policy.keep_tables_intact:
            add_block_text(block, block.text.strip())
            flush_chunk()
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


__all__ = ["Chunk", "ChunkerResult", "apply_plan"]
