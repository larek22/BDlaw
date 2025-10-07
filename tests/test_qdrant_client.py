import uuid

import pytest

pytest.importorskip("qdrant_client")

from app.chunker import Chunk
from app.qdrant_client import _make_point_id


def test_make_point_id_returns_deterministic_uuid():
    chunk = Chunk(
        doc_id="doc",
        path="/tmp/doc",
        page=1,
        chunk_index=3,
        offset=0,
        text="hello",
        sha="a" * 64,
    )

    point_id = _make_point_id(chunk)

    # ensure valid uuid string
    parsed = uuid.UUID(point_id)
    assert str(parsed) == point_id

    # ensure deterministic for same chunk
    assert point_id == _make_point_id(chunk)

    # ensure different chunk index changes id
    different = Chunk(
        doc_id="doc",
        path="/tmp/doc",
        page=1,
        chunk_index=4,
        offset=0,
        text="hello",
        sha="a" * 64,
    )
    assert _make_point_id(different) != point_id
