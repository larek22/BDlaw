import pytest

pytest.importorskip("qdrant_client")

from app.chunker import Chunk
from app.qdrant_client import _make_point_id, _vector_size_for_model, _mask_api_key


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

    point_id = _make_point_id(
        chunk,
        embedding_model="text-embedding-3-large",
        chunk_size=1200,
        chunk_overlap=120,
    )

    assert isinstance(point_id, str)
    assert len(point_id) == 36  # UUID string length with hyphens
    assert point_id == _make_point_id(
        chunk,
        embedding_model="text-embedding-3-large",
        chunk_size=1200,
        chunk_overlap=120,
    )

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
    different_id = _make_point_id(
        different,
        embedding_model="text-embedding-3-large",
        chunk_size=1200,
        chunk_overlap=120,
    )
    assert different_id != point_id


def test_vector_size_for_model_defaults():
    assert _vector_size_for_model("text-embedding-3-large") == 3072
    assert _vector_size_for_model("anything-else") == 1536


def test_mask_api_key():
    assert _mask_api_key(None) == "<none>"
    assert _mask_api_key("") == "<none>"
    assert _mask_api_key("abcd") == "***"
    assert _mask_api_key("abcdefgh") == "ab...gh"
