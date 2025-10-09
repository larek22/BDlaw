from __future__ import annotations

import pytest

pytest.importorskip("qdrant_client")

from app.qdrant_client import _mask_api_key, _payload_schema, _vector_size_for_model


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
