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


class _FakeChatCompletions:
    def create(self, *args, **kwargs):
        raise PermissionDeniedError(message="denied", response=None, body=None)


class _FakeChat:
    def __init__(self):
        self.completions = _FakeChatCompletions()


class _FakeOpenAI:
    def __init__(self, *args, **kwargs):
        self.chat = _FakeChat()


def test_query_pipeline_permission_error(monkeypatch):
    monkeypatch.setattr("app.query_pipeline.OpenAI", _FakeOpenAI)
    settings = AppSettings()
    pipeline = QueryPipeline(settings, DummyEmbeddingClient(), DummyVectorStore())

    with pytest.raises(RuntimeError) as exc:
        pipeline.answer("What is the answer?")

    assert "denied access" in str(exc.value)
