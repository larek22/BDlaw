import pytest

openai = pytest.importorskip("openai")
from openai import PermissionDeniedError  # type: ignore  # noqa: E402

from app.query_pipeline import QueryPipeline
from app.settings import AppSettings


class DummyEmbeddingClient:
    def embed_query(self, text: str) -> list[float]:
        return [0.0]


class DummyVectorStore:
    def search(self, vector: list[float], top_k: int):
        return []


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
