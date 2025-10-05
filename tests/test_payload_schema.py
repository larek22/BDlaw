from datetime import UTC, datetime, date

import pytest

from bdlaw.index.schema import NormPayload


def _payload_kwargs(**overrides):
    base = dict(
        jurisdiction="ru",
        act_type="federal_law",
        division="Подраздел 1",
        chapter="Глава 1",
        section="Раздел 1",
        law_code="ГК РФ",
        law_full_title="Гражданский кодекс",
        article="12",
        part="1",
        point=None,
        subpoint=None,
        version_id="2024-01-01",
        valid_from=date(2024, 1, 1),
        valid_to=None,
        supersedes=None,
        source_url="https://example.com",
        citation="ГК РФ, ст. 12 (ред. 2024-01-01)",
        lang="ru",
        raw_text="Текст",
        clean_text="Текст",
        annotations=[],
        amendments=[],
        cross_refs=[],
        page_spans=[(1, 1)],
        span_offsets=[(0, 10)],
        tokens_est=10,
        file_origin="/tmp/doc.pdf",
        ingested_at=datetime.now(UTC),
    )
    base.update(overrides)
    return base


def test_norm_payload_builds_hash_and_gid():
    payload = NormPayload.build(**_payload_kwargs())
    assert len(payload.gid) == 64
    assert len(payload.hash) == 64


def test_norm_payload_rejects_hash_mismatch():
    with pytest.raises(ValueError):
        NormPayload(
            gid="0" * 64,
            hash="1" * 64,
            **_payload_kwargs(clean_text="Один", raw_text="Другой"),
        )
