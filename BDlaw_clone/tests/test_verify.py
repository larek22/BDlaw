from types import SimpleNamespace

from verify import _check_reference_queries, _record_total_count


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
    chunk = SimpleNamespace(doc_id="gkrf:part4:art1283:v2006-12-18", chunk_index=0)
    source = SimpleNamespace(chunk=chunk, score=0.5)
    pipeline = DummyPipeline({"исключительное право": [source]})

    failures, info = _check_reference_queries(
        pipeline,
        ["gkrf:part4:art1229:v2006-12-18"],
    )

    assert info  # other queries skipped
    assert any("исключительное право" in failure for failure in failures)
    assert any("gkrf:part4:art1229" in failure for failure in failures)


def test_record_total_count_success():
    ok, message = _record_total_count(total_points=10, expected_points=10)

    assert ok is True
    assert "matches" in message


def test_record_total_count_mismatch_warning():
    ok, message = _record_total_count(total_points=12, expected_points=10)

    assert ok is False
    assert "mismatch" in message
