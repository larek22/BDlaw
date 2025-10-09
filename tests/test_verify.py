from app.legal_types import ChunkRecord
from app.query_pipeline import SourceChunk

from verify import _check_reference_queries


class DummyPipeline:
    def __init__(self, responses=None):
        self.responses = responses or {}
        self.calls = []

    def retrieve(self, question, top_k=5):
        self.calls.append((question, top_k))
        return self.responses.get(question, [])


def test_reference_queries_skipped_when_expected_docs_missing():
    pipeline = DummyPipeline()
    failures, info = _check_reference_queries(pipeline, [])

    assert failures == []
    assert info  # skip messages are recorded
    assert pipeline.calls == []  # retrieval not attempted


def test_reference_query_failure_uses_available_expectations():
    chunk = ChunkRecord(
        doc_id="gkrf:part4:art1283:v2006-12-18",
        chunk_index=0,
        chunk_id="chunk-0",
        title_text="",
        body_text="",
        hierarchy={},
        law_meta={},
        source={},
        chunk_sha256="body",
        title_sha256="title",
        body_sha256="body",
    )
    source = SourceChunk(chunk=chunk, score=0.5)
    pipeline = DummyPipeline({"исключительное право": [source]})

    failures, info = _check_reference_queries(
        pipeline,
        ["gkrf:part4:art1229:v2006-12-18"],
    )

    assert info  # other queries skipped
    assert any("исключительное право" in failure for failure in failures)
    assert any("gkrf:part4:art1229" in failure for failure in failures)
