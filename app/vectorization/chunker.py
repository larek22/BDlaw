"""Deterministic chunker that applies a :class:`VectorizationPlan`."""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, TYPE_CHECKING

from .heading_index import build_heading_index
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
    heading_index = build_heading_index(blocks, plan)

    chunks: List[Chunk] = []
    window_texts: List[str] = []
    window_tokens: List[int] = []
    window_meta: List[Dict[str, object]] = []
    window_token_total = 0

    token_usage = 0
    chunk_index = 0
    last_start = -1
    last_end = -1

    source_path = blocks[0].path if blocks else ""
    source_basename = Path(source_path).stem if source_path else "document"

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
        overlap_text = tokenizer.decode(overlap_slice)
        window_texts = [overlap_text]
        window_tokens = [len(overlap_slice)]
        window_meta = list(meta_list)
        window_token_total = len(overlap_slice)

    def _fallback_title(index: int) -> str:
        return f"{source_basename} fragment {index + 1}"

    def _emit_chunk(text: str, meta_list: Sequence[Dict[str, object]]) -> Chunk | None:
        nonlocal chunk_index, last_start, last_end, token_usage

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

        chunk_id = _make_chunk_id(
            plan,
            start=start_pos,
            end=end_pos,
            index=chunk_index,
            source=source_path,
        )

        doc_id_value: str | None = None
        for meta in meta_list:
            if not isinstance(meta, dict):
                continue
            if meta.get("doc_id"):
                candidate = str(meta.get("doc_id") or "").strip()
                if candidate:
                    doc_id_value = candidate
                    break
            attrs = meta.get("attrs")
            if isinstance(attrs, dict) and attrs.get("doc_id"):
                candidate = str(attrs.get("doc_id") or "").strip()
                if candidate:
                    doc_id_value = candidate
                    break
        if not doc_id_value:
            doc_id_value = source_basename
        doc_id_value = (doc_id_value or "document").strip() or "document"
        if len(doc_id_value) > 512:
            doc_id_value = doc_id_value[:512]

        chunk_meta: Dict[str, object] = {
            "source_file": source_path,
            "index": chunk_index,
            "doc_id": doc_id_value,
        }

        hierarchy_meta = heading_index.hierarchy_for(start_pos)
        if hierarchy_meta:
            chunk_meta["hierarchy"] = hierarchy_meta
            for field in plan.hierarchy_rules.path_fields:
                value = hierarchy_meta.get(field)
                if value is not None:
                    chunk_meta[field] = value
            for extra_key in (
                "section_roman",
                "section_title",
                "part_no",
                "part_title",
                "chapter_no",
                "chapter_title",
                "article_no",
                "article_no_str",
                "article_no_int",
                "article_suffix",
                "article_title",
            ):
                if extra_key in hierarchy_meta:
                    chunk_meta[extra_key] = hierarchy_meta[extra_key]

        if heading_index.language:
            chunk_meta.setdefault("lang", heading_index.language)

        row_indices = [
            meta.get("attrs", {}).get("row_index")
            for meta in meta_list
            if isinstance(meta, dict)
            and isinstance(meta.get("attrs"), dict)
            and isinstance(meta.get("attrs", {}).get("row_index"), int)
        ]
        if row_indices:
            chunk_meta["row_index"] = row_indices[-1]

        title_text = heading_index.title_for(
            start_pos,
            cleaned_text,
            fallback=_fallback_title(chunk_index),
        )
        chunk_meta["title_text"] = title_text

        chunk = Chunk(
            text=cleaned_text,
            tokens=token_count,
            start=start_pos,
            end=end_pos,
            meta=chunk_meta,
            chunk_id=chunk_id,
        )
        chunk.meta["plan_chunk_id"] = chunk_id
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

        combined_text = "".join(previous_texts)
        if not combined_text.strip() and plan.quality_checks.forbid_empty_text:
            return

        segments = [combined_text]
        if total_tokens > policy.max_tokens:
            segments = _split_long_text(combined_text, policy, tokenizer)

        last_chunk_emitted: Chunk | None = None
        for segment in segments:
            emitted = _emit_chunk(segment, previous_meta)
            if emitted is not None:
                last_chunk_emitted = emitted

        if not final and last_chunk_emitted is not None:
            _append_overlap(last_chunk_emitted, previous_meta)

    def add_block_text(block: Block, text: str, *, allow_newline: bool = True) -> None:
        nonlocal window_token_total, token_usage
        if not text:
            return

        trimmed = text.strip()
        if not trimmed and plan.quality_checks.forbid_empty_text:
            return

        block_meta = {
            "start": (block.position or {}).get("start", 0),
            "end": (block.position or {}).get("end", 0),
            "type": block.type,
            "attrs": dict(block.attrs or {}),
            "path": block.path,
        }

        block_tokens = tokenizer.count(trimmed)
        if block_tokens > policy.max_tokens:
            flush_chunk()
            for slice_text in _split_long_text(trimmed, policy, tokenizer):
                token_usage += tokenizer.count(slice_text)
                emitted = _emit_chunk(slice_text, [block_meta])
                if emitted is not None:
                    _append_overlap(emitted, [block_meta])
            return

        prefix = "\n" if window_texts and allow_newline else ""
        segment = f"{prefix}{trimmed}" if trimmed else prefix
        segment_tokens = tokenizer.count(segment)
        if segment_tokens == 0 and plan.quality_checks.forbid_empty_text:
            return

        if window_texts and window_token_total + segment_tokens > policy.max_tokens:
            flush_chunk()
            prefix = ""
            segment = trimmed
            segment_tokens = tokenizer.count(segment)
            if segment_tokens == 0 and plan.quality_checks.forbid_empty_text:
                return

        window_texts.append(segment)
        window_tokens.append(segment_tokens)
        window_meta.append(block_meta)
        window_token_total += segment_tokens
        token_usage += segment_tokens

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
                add_block_text(buffer[-1], record_text, allow_newline=False)
                flush_chunk()
                buffer = []
        if buffer:
            record_text = "\n".join(
                _record_payload_text(item, record_policy) for item in buffer
            )
            add_block_text(buffer[-1], record_text, allow_newline=False)
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

    heading_regex = plan.hierarchy_rules.heading_regex or {}

    for block in blocks:
        if block.type == "heading":
            level = int(block.attrs.get("level", 1)) if isinstance(block.attrs, dict) else 1
            heading_text = block.text.strip()
            should_split = level in split_levels
            if split_names and heading_text:
                for name in split_names:
                    pattern = heading_regex.get(name)
                    if pattern and re.search(pattern, heading_text):
                        should_split = True
                        break
            if should_split and window_texts:
                flush_chunk()
            if heading_text:
                add_block_text(block, heading_text, allow_newline=False)
                flush_chunk()
            continue

        if block.type == "list_item" and policy.keep_lists_intact:
            add_block_text(block, f"- {block.text.strip()}")
            continue

        if block.type == "table" and policy.keep_tables_intact:
            add_block_text(block, block.text.strip(), allow_newline=False)
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
