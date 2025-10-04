from datetime import UTC, datetime, date

from bdlaw.index.pipeline import IndexingPipeline
from bdlaw.index.schema import NormPayload
from bdlaw.settings.config import ChunkingConfig


class DummyEmbedder:
    def __init__(self):
        self.calls = []

    def embed_batch(self, texts):
        self.calls.append(list(texts))
        return [[0.0] * 3 for _ in texts]


class DummyStore:
    def __init__(self):
        self.upserts = []

    def upsert(self, norms, vectors):
        self.upserts.append((list(norms), list(vectors)))


def _norm(article: str, text: str) -> NormPayload:
    return NormPayload.build(
        jurisdiction="ru",
        act_type="federal_law",
        division=None,
        chapter=None,
        section=None,
        law_code="ГК РФ",
        law_full_title="Гражданский кодекс",
        article=article,
        part=None,
        point=None,
        subpoint=None,
        version_id="2024-01-01",
        valid_from=date(2024, 1, 1),
        valid_to=None,
        supersedes=None,
        source_url="https://example.com",
        citation=f"ГК РФ, ст. {article} (ред. 2024-01-01)",
        lang="ru",
        raw_text=text,
        clean_text=text,
        annotations=[],
        amendments=[],
        cross_refs=[],
        page_spans=[],
        span_offsets=[(0, len(text))],
        tokens_est=len(text.split()),
        file_origin=None,
        ingested_at=datetime.now(UTC),
    )


def test_indexing_pipeline_deduplicates_by_hash():
    embedder = DummyEmbedder()
    store = DummyStore()
    pipeline = IndexingPipeline(embedder, store, ChunkingConfig())
    norm_a = _norm("12", "Текст")
    norm_b = _norm("12", "Текст")
    stats = pipeline.index([norm_a, norm_b])
    assert stats.norms_indexed == 1
    assert len(store.upserts[0][0]) == 1
    assert len(embedder.calls) == 1
