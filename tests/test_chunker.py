from datetime import UTC, datetime

from bdlaw.index.schema import NormPayload
from bdlaw.parser.chunker import chunk_norm
from bdlaw.settings.config import ChunkingConfig


def _norm_with_span(text: str, start: int) -> NormPayload:
    return NormPayload.build(
        jurisdiction="ru",
        act_type="federal_law",
        division=None,
        chapter=None,
        section=None,
        law_code="ГК РФ",
        law_full_title="Гражданский кодекс",
        article="12",
        part="1",
        point="1",
        subpoint=None,
        version_id="2024-01-01",
        valid_from=datetime(2024, 1, 1, tzinfo=UTC).date(),
        valid_to=None,
        supersedes=None,
        source_url="https://example.com",
        citation="ГК РФ, ст. 12, п. 1 (ред. 2024-01-01)",
        lang="ru",
        raw_text=text,
        clean_text=text,
        annotations=[],
        amendments=[],
        cross_refs=[],
        page_spans=[],
        span_offsets=[(start, start + len(text))],
        tokens_est=len(text.split()),
        file_origin=None,
        ingested_at=datetime.now(UTC),
    )


def test_chunk_norm_preserves_span_offsets():
    text = "Первое предложение. Второе предложение. Третье предложение."
    norm = _norm_with_span(text, 200)
    config = ChunkingConfig(max_words=4, overlap_words=1)
    chunks = chunk_norm(norm, config)
    assert len(chunks) > 1
    # spans cover the original range without gaps
    total_start = chunks[0].span_offsets[0][0]
    total_end = chunks[-1].span_offsets[0][1]
    assert total_start == norm.span_offsets[0][0]
    assert total_end == norm.span_offsets[0][1]
    for chunk in chunks:
        start, end = chunk.span_offsets[0]
        assert norm.span_offsets[0][0] <= start < end <= norm.span_offsets[0][1]
        assert chunk.clean_text == chunk.raw_text
