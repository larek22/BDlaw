from __future__ import annotations

import pytest

pytest.importorskip("qdrant_client")

from types import SimpleNamespace

from app.qdrant_client import (
    QdrantVectorStore,
    _collection_matches,
    _mask_api_key,
    _payload_schema,
    _vector_size_for_model,
)
from app.utils.net import normalize_qdrant_url


def test_normalize_qdrant_url_port_and_scheme():
    assert normalize_qdrant_url(None) == "http://localhost:6333"
    assert normalize_qdrant_url("localhost") == "http://localhost:6333"
    assert normalize_qdrant_url("https://demo.qdrant.io") == "https://demo.qdrant.io:6333"


def test_vector_size_for_model_defaults():
    assert _vector_size_for_model("text-embedding-3-large") == 3072
    assert _vector_size_for_model("unknown-model") == 1536


def test_mask_api_key_formats():
    assert _mask_api_key(None) == "<none>"
    assert _mask_api_key("abcd") == "***"
    assert _mask_api_key("abcdefgh") == "ab...gh"


def test_payload_schema_fallback():
    from app import qdrant_client as qc  # local import to access module object

    original = getattr(qc.rest, "PayloadSchemaType", None)
    try:
        class DummyEnum:
            KEYWORD = "keyword"

        setattr(qc.rest, "PayloadSchemaType", DummyEnum)
        assert _payload_schema("keyword") == "keyword"
        delattr(qc.rest, "PayloadSchemaType")
        assert _payload_schema("text") == "text"
    finally:
        if original is not None:
            setattr(qc.rest, "PayloadSchemaType", original)


class _DummyVector:
    def __init__(self, size: int) -> None:
        self.size = size


class _DummyInfo:
    def __init__(self, vectors) -> None:
        self.config = type("Cfg", (), {"params": type("Params", (), {"vectors": vectors})})


def test_collection_matches_with_named_vectors():
    vectors = {"title_vec": _DummyVector(3072), "body_vec": _DummyVector(3072)}
    info = _DummyInfo(vectors)
    assert _collection_matches(info, 3072)


def test_collection_matches_rejects_single_vector():
    info = _DummyInfo(_DummyVector(3072))
    assert not _collection_matches(info, 3072)


def test_collection_matches_rejects_wrong_names():
    vectors = {"default": _DummyVector(3072)}
    info = _DummyInfo(vectors)
    assert not _collection_matches(info, 3072)


def _make_store(client):
    store = object.__new__(QdrantVectorStore)
    store._client = client
    store._alias_name = "kb_docs_v1"
    store._write_collection_name = "kb_docs_v1"
    return store


def test_payload_to_chunk_coerces_strings():
    store = _make_store(SimpleNamespace())
    payload = {
        "doc_id": 123,
        "chunk_index": "5",
        "chunk_id": 456,
        "title_text": None,
        "body_text": "Текст",
        "hierarchy": {"chapter_no": 7},
        "law_meta": None,
        "source": None,
    }

    chunk = store.payload_to_chunk(payload)

    assert chunk.doc_id == "123"
    assert chunk.chunk_index == 5
    assert chunk.title_text.startswith("Текст")
    assert chunk.hierarchy["chapter_no"] == 7


def test_list_chapters_collects_unique_values():
    batches = [
        (
            [
                SimpleNamespace(payload={"hierarchy": {"chapter_no": 1}}),
                SimpleNamespace(payload={"hierarchy": {"chapter_no": 2}}),
            ],
            None,
        )
    ]

    class DummyClient:
        def scroll(self, **kwargs):
            return batches.pop(0) if batches else ([], None)

    store = _make_store(DummyClient())

    assert store.list_chapters() == [1, 2]


def test_fetch_article_chunks_sorts_by_chunk_index():
    payloads = [
        {"doc_id": "doc", "chunk_index": 4, "title_text": "", "body_text": "b"},
        {"doc_id": "doc", "chunk_index": 1, "title_text": "", "body_text": "a"},
    ]
    batches = [
        ([SimpleNamespace(payload=payload) for payload in payloads], None)
    ]

    class DummyClient:
        def scroll(self, **kwargs):
            return batches.pop(0) if batches else ([], None)

    store = _make_store(DummyClient())
    store.build_doc_id_filter = lambda doc_ids: {"doc_ids": doc_ids}

    records = store.fetch_article_chunks("doc")

    assert [record.chunk_index for record in records] == [1, 4]


def test_active_collection_target_uses_alias_resolution():
    store = _make_store(SimpleNamespace())
    store._resolve_alias = lambda alias: "physical"  # type: ignore[attr-defined]

    assert store.active_collection_target() == "physical"
