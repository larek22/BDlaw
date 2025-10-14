"""Deterministic chunker that applies a :class:`VectorizationPlan`."""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
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


def _build_overlap_window(
    texts: Sequence[str],
    tokens: Sequence[int],
    meta: Sequence[Dict[str, object]],
    overlap_tokens: int,
    tokenizer: Tokenizer,
) -> tuple[List[str], List[int], List[Dict[str, object]]]:
    if overlap_tokens <= 0:
        return [], [], []

    collected_texts: List[str] = []
    collected_tokens: List[int] = []
    collected_meta: List[Dict[str, object]] = []
    total = 0

    for text_piece, token_count, meta_piece in zip(
        reversed(texts), reversed(tokens), reversed(meta)
    ):
        collected_texts.insert(0, text_piece)
        collected_tokens.insert(0, token_count)
        collected_meta.insert(0, meta_piece)
        total += token_count
        if total >= overlap_tokens:
            break

    if total <= overlap_tokens:
        return collected_texts, collected_tokens, collected_meta

    excess = total - overlap_tokens
    idx = 0
    while excess > 0 and idx < len(collected_texts):
        token_count = collected_tokens[idx]
        if token_count <= excess:
            excess -= token_count
            collected_texts.pop(idx)
            collected_tokens.pop(idx)
            collected_meta.pop(idx)
            continue

        keep_tokens = token_count - excess
        if keep_tokens <= 0:
            collected_texts.pop(idx)
            collected_tokens.pop(idx)
            collected_meta.pop(idx)
            excess -= token_count
            continue

        encoded = tokenizer.encode(collected_texts[idx])
        slice_tokens = encoded[-keep_tokens:]
        collected_texts[idx] = tokenizer.decode(slice_tokens)
        collected_tokens[idx] = len(slice_tokens)
        excess = 0

    return collected_texts, collected_tokens, collected_meta


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
    last_start = -1
    last_end = -1

    def flush_chunk(final: bool = False) -> None:
        nonlocal window_texts, window_tokens, window_token_total, chunk_index, window_meta, last_start, last_end

        if not window_texts:
            return

        previous_texts = list(window_texts)
        previous_tokens = list(window_tokens)
        previous_meta = list(window_meta)

        window_texts = []
        window_tokens = []
        window_meta = []
        window_token_total = 0

        text = "\n".join(previous_texts).strip()
        if not text and plan.quality_checks.forbid_empty_text:
            window_texts = []
            window_tokens = []
            window_token_total = 0
            return
        token_count = sum(previous_tokens)
        if token_count > policy.max_tokens:
            raise RuntimeError(
                f"Chunk tokens {token_count} exceed max {policy.max_tokens}"
            )
        start_pos = 0
        end_pos = 0
        if previous_meta:
            start_pos = min(
                meta.get("start", 0) for meta in previous_meta if isinstance(meta, dict)
            )
            end_pos = max(
                meta.get("end", 0) for meta in previous_meta if isinstance(meta, dict)
            )
        if chunk_index > 0:
            if start_pos <= last_start:
                start_pos = last_start + 1
            if end_pos <= start_pos:
                end_pos = start_pos + 1
            if end_pos <= last_end:
                end_pos = last_end + 1
        last_start = start_pos
        last_end = end_pos
        chunk_id = _make_chunk_id(
            plan,
            start=start_pos,
            end=end_pos,
            index=chunk_index,
            source=blocks[0].path if blocks else "",
        )
        chunk_meta: Dict[str, object] = {
            "source_file": blocks[0].path if blocks else "",
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
            or heading_values[-1]
            if heading_values
            else ""
        )
        if title_candidate:
            chunk_meta["title_text"] = title_candidate
        row_indices = [
            meta.get("attrs", {}).get("row_index")
            for meta in previous_meta
            if isinstance(meta.get("attrs", {}).get("row_index"), int)
        ]
        if row_indices:
            chunk_meta["row_index"] = row_indices[-1]
        source_path = blocks[0].path if blocks else ""
        if source_path and "doc_id" not in chunk_meta:
            chunk_meta["doc_id"] = Path(source_path).stem
        chunks.append(
            Chunk(
                text=text,
                tokens=token_count,
                start=start_pos,
                end=end_pos,
                meta=chunk_meta,
                chunk_id=chunk_id,
            )
        )
        chunk_meta["plan_chunk_id"] = chunk_id
        chunk_index += 1

        if final or overlap_tokens <= 0:
            return

        (
            window_texts,
            window_tokens,
            window_meta,
        ) = _build_overlap_window(
            previous_texts,
            previous_tokens,
            previous_meta,
            overlap_tokens,
            tokenizer,
        )
        window_token_total = sum(window_tokens)

    window_meta: List[Dict[str, object]] = []
    token_usage = 0
    named_hierarchy: Dict[str, str] = {}

    def add_block_text(block: Block, text: str) -> None:
        nonlocal window_token_total, token_usage
        if not text:
            return
        tokens = tokenizer.count(text)
        if tokens == 0 and plan.quality_checks.forbid_empty_text:
            return
        token_usage += tokens
        if tokens > policy.max_tokens:
            slices = _split_long_text(
                text,
                tokenizer,
                policy.max_tokens,
                policy.overlap_tokens,
            )
            for slice_text in slices:
                add_block_text(block, slice_text)
            return
        if window_token_total + tokens > policy.max_tokens and window_texts:
            flush_chunk()
        window_texts.append(text)
        window_tokens.append(tokens)
        block_meta = {
            "start": (block.position or {}).get("start", 0),
            "end": (block.position or {}).get("end", 0),
            "type": block.type,
            "attrs": dict(block.attrs or {}),
            "text": block.text,
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


def _split_long_text(
    text: str,
    tokenizer: Tokenizer,
    max_tokens: int,
    overlap_tokens: int,
) -> List[str]:
    if not text:
        return []
    if max_tokens <= 0:
        return [text]

    tokens = tokenizer.encode(text)
    if not tokens:
        return [text]

    overlap = max(0, min(overlap_tokens, max_tokens - 1))
    step = max(1, max_tokens - overlap)

    slices: List[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        window = tokens[start:end]
        if not window:
            break
        slice_text = tokenizer.decode(window)
        # Guard against tokenizer decode/encode drift inflating token count
        slice_tokens = tokenizer.encode(slice_text)
        while len(slice_tokens) > max_tokens and len(window) > 1:
            window = window[:-1]
            slice_text = tokenizer.decode(window)
            slice_tokens = tokenizer.encode(slice_text)
        slices.append(slice_text)
        if end >= len(tokens):
            break
        start += step

    cleaned = [slice_text for slice_text in slices if slice_text]
    return cleaned or [text]


__all__ = ["Chunk", "ChunkerResult", "apply_plan"]
