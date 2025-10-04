from types import SimpleNamespace

from bdlaw.search.service import SearchResult
from bdlaw.testsuite.runner import GoldenQuery, TestSuiteRunner


class DummySearch:
    def __init__(self, results):
        self._results = results

    def search(self, **kwargs):
        return self._results


class FakeValidator:
    def validate(self, answer, citations, context_citations):
        class Result:
            verdict = "PASS"
            explanation = "ok"

        return Result()


def test_testsuite_metrics():
    results = [
        SearchResult(
            citation="ГК РФ ст. 12",
            score=1.0,
            payload={
                "law_code": "ГК РФ",
                "article": "12",
                "part": None,
                "citation": "ГК РФ ст. 12",
                "source_url": "https://example.com",
            },
        ),
        SearchResult(
            citation="ГК РФ ст. 10",
            score=0.8,
            payload={
                "law_code": "ГК РФ",
                "article": "10",
                "part": None,
                "citation": "ГК РФ ст. 10",
                "source_url": "https://example.com/10",
            },
        ),
    ]
    runner = TestSuiteRunner(DummySearch(results), k=2, answer_validator=FakeValidator())
    report = runner.run(
        [
            GoldenQuery(
                query="Как защищать права?",
                expected_norms=[{"law_code": "ГК РФ", "article": "12", "part": None}],
            )
        ]
    )
    assert report.recall_at_k == 1.0
    assert report.precision_at_k == 0.5
    assert report.mrr_at_k == 1.0
    assert report.ndcg_at_k > 0
    assert report.answer_pass_at_k == 1.0


def test_runner_initializes_validator_from_model(monkeypatch):
    created = {}

    class DummyAnswerValidator:
        def __init__(self, model):
            created["model"] = model

        def validate(self, answer, citations, context_citations):
            return SimpleNamespace(verdict="PASS", explanation="ok")

    monkeypatch.setattr("bdlaw.gpt.validator.AnswerValidator", DummyAnswerValidator)

    results = [
        SearchResult(
            citation="ГК РФ ст. 12",
            score=1.0,
            payload={"law_code": "ГК РФ", "article": "12", "part": None, "citation": "ГК РФ ст. 12"},
        )
    ]
    runner = TestSuiteRunner(DummySearch(results), k=1, judge_model="dummy-model")
    report = runner.run(
        [GoldenQuery(query="", expected_norms=[{"law_code": "ГК РФ", "article": "12", "part": None}])]
    )
    assert created["model"] == "dummy-model"
    assert report.answer_pass_at_k == 1.0
