import pytest

openai = pytest.importorskip("openai")
from openai import PermissionDeniedError  # type: ignore  # noqa: E402

from types import SimpleNamespace

from qdrant_client.http import models as rest

from app.query_pipeline import QueryPipeline
from app.settings import AppSettings


class DummyEmbeddingClient:
    def embed_query(self, text: str) -> list[float]:
        return [0.0, 0.0, 0.0]


class DummyVectorStore:
    def __init__(self) -> None:
        self.query_calls: list[tuple[str, int]] = []
        self.prefilter_calls: list[tuple[str, int]] = []
        self.keyword_prefilter_results: list[str] = []

    def query(self, *, vector_name: str, query_vector: list[float], limit: int, filters=None):
        self.query_calls.append((vector_name, limit))
        return []

    def keyword_prefilter(self, query: str, limit: int) -> list[str]:
        self.prefilter_calls.append((query, limit))
        return list(self.keyword_prefilter_results)

    def build_doc_id_filter(self, doc_ids):  # pragma: no cover - not used in assertions
        return doc_ids

    def build_as_of_filter(self, *, status=None, as_of_start=None, as_of_date=None):  # pragma: no cover - deterministic stub
        return {"status": status, "start": as_of_start, "end": as_of_date}

    def combine_filters(self, *filters):  # pragma: no cover - deterministic stub
        return [flt for flt in filters if flt]


class _Choice:
    def __init__(self, content: str):
        self.message = type("Message", (), {"content": content})()


class _Response:
    def __init__(self, content: str):
        self.choices = [_Choice(content)]


class _FallbackChatCompletions:
    def __init__(self):
        self.calls: list[tuple[str, int | None]] = []

    def create(self, *, model: str, messages, temperature: float, max_tokens: int | None = None):  # type: ignore[override]
        self.calls.append((model, max_tokens))
        if model in {"gpt-4.1-nano", "gpt-4.1-mini"}:
            raise PermissionDeniedError(message="denied", response=None, body=None)
        if model == "gpt-4o-mini" and max_tokens == 1:
            return _Response("probe")
        if model == "gpt-4o-mini":
            return _Response("Fallback answer")
        raise AssertionError(f"unexpected model {model}")


class _AlwaysDeniedCompletions:
    def create(self, *, model: str, messages, temperature: float, max_tokens: int | None = None):  # type: ignore[override]
        raise PermissionDeniedError(message="denied", response=None, body=None)


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeOpenAI:
    def __init__(self, completions):
        self.chat = _FakeChat(completions)


def test_query_pipeline_permission_error_auto_fallback(monkeypatch, tmp_path):
    completions = _FallbackChatCompletions()

    def _factory(*args, **kwargs):
        return _FakeOpenAI(completions)

    monkeypatch.setattr("app.query_pipeline.OpenAI", _factory)
    monkeypatch.setattr("app.settings.CONFIG_DIR", tmp_path, raising=False)
    monkeypatch.setattr("app.settings.CONFIG_PATH", tmp_path / "config.json", raising=False)

    settings = AppSettings()
    settings.openai_models.chat = "gpt-4.1-nano"
    pipeline = QueryPipeline(settings, DummyEmbeddingClient(), DummyVectorStore())

    result = pipeline.answer("What is the answer?")

    assert result.answer == "Fallback answer"
    assert settings.openai_models.chat == "gpt-4o-mini"
    assert completions.calls == [
        ("gpt-4.1-nano", 1),
        ("gpt-4.1-mini", 1),
        ("gpt-4o-mini", 1),
        ("gpt-4o-mini", None),
    ]
    assert (tmp_path / "config.json").exists()


def test_query_pipeline_permission_error_both_fail(monkeypatch):
    def _factory(*args, **kwargs):
        return _FakeOpenAI(_AlwaysDeniedCompletions())

    monkeypatch.setattr("app.query_pipeline.OpenAI", _factory)

    settings = AppSettings()
    settings.openai_models.chat = "gpt-4.1-nano"
    pipeline = QueryPipeline(settings, DummyEmbeddingClient(), DummyVectorStore())

    with pytest.raises(RuntimeError) as exc:
        pipeline.answer("What is the answer?")

    assert "No accessible OpenAI chat model" in str(exc.value)


def test_tokenize_query_filters_stopwords_and_handles_empty(monkeypatch):
    monkeypatch.setattr("app.query_pipeline.pymorphy2", None, raising=False)
    settings = AppSettings()
    pipeline = QueryPipeline(settings, DummyEmbeddingClient(), DummyVectorStore())

    assert pipeline._tokenize_query("") == []
    tokens = pipeline._tokenize_query("И Исключительное право")
    assert tokens == ["исключительное", "право"]


def test_build_chapter_filter_and_article_range_filters(monkeypatch):
    monkeypatch.setattr("app.query_pipeline.pymorphy2", None, raising=False)
    settings = AppSettings()
    pipeline = QueryPipeline(settings, DummyEmbeddingClient(), DummyVectorStore())

    assert pipeline._build_chapter_filter("all") is None
    chapter_filter = pipeline._build_chapter_filter("73")
    assert isinstance(chapter_filter, rest.Filter)
    assert chapter_filter.must[0].match.value == 73

    if not hasattr(rest, "Range"):
        pytest.skip("rest.Range not available in current qdrant-client build")

    article_filter = pipeline._build_article_range_filter("1446.1", "1450")
    assert isinstance(article_filter, rest.Filter)
    condition = article_filter.must[0]
    assert condition.range.gte == 1446
    assert condition.range.lte == 1450

    assert pipeline._build_article_range_filter(None, None) is None


def test_keyword_boost_changes_ranking(monkeypatch):
    monkeypatch.setattr("app.query_pipeline.pymorphy2", None, raising=False)
    settings = AppSettings()
    settings.query.keyword_boost_weight = 0.5
    pipeline = QueryPipeline(settings, DummyEmbeddingClient(), DummyVectorStore())

    good_payload = {"title_text": "Исключительное право", "body_text": ""}
    neutral_payload = {"title_text": "Другое", "body_text": ""}
    boosted = SimpleNamespace(id="1", payload=good_payload, score=0.1)
    neutral = SimpleNamespace(id="2", payload=neutral_payload, score=0.1)

    query_terms = pipeline._tokenize_query("исключительное право")
    fused = pipeline._fuse_results(
        body_results=[neutral, boosted],
        title_results=[],
        body_weight=1.0,
        title_weight=0.0,
        rrf_k=60,
        query_terms=query_terms,
    )

    assert fused[0].payload == good_payload


def test_build_combined_filter_applies_prefilter(monkeypatch):
    monkeypatch.setattr("app.query_pipeline.pymorphy2", None, raising=False)
    settings = AppSettings()
    vector_store = DummyVectorStore()
    vector_store.keyword_prefilter_results = ["doc-1"]
    pipeline = QueryPipeline(settings, DummyEmbeddingClient(), vector_store)

    combined = pipeline._build_combined_filter("query", top_k=5)

    assert vector_store.prefilter_calls
    assert combined  # prefilter + temporal filter
