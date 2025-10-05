from typing import List

from qdrant_client.http import models as rest

from bdlaw.search.service import SearchService


class DummyEmbedder:
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        return [[0.0] * 3 for _ in texts]


class DummyStore:
    def __init__(self):
        self.filters = []
        self.responses: List[List[rest.ScoredPoint]] = []

    def upsert(self, norms, vectors):
        raise NotImplementedError

    def search(self, query, k, filters=None):
        self.filters.append(filters)
        if self.responses:
            return self.responses.pop(0)
        return []


def _scored_point(gid: str, citation: str, article: str, part: str | None, cross_refs=None) -> rest.ScoredPoint:
    return rest.ScoredPoint(
        id=gid,
        score=1.0,
        payload={
            "gid": gid,
            "citation": citation,
            "law_code": "ГК РФ",
            "article": article,
            "part": part,
            "source_url": "https://example.com",
            "cross_refs": cross_refs or [],
            "valid_from": "2020-01-01",
            "valid_to": None,
        },
        version=0,
    )


def test_build_filter_with_date_and_law_code():
    store = DummyStore()
    store.responses.append([
        _scored_point(
            "1",
            "ГК РФ ст. 12",
            "12",
            None,
            cross_refs=[{"law_code": "ГК РФ", "article": "395", "part": "1"}],
        )
    ])
    store.responses.append([
        _scored_point("2", "ГК РФ ст. 395", "395", "1"),
    ])
    service = SearchService(DummyEmbedder(), store)  # type: ignore[arg-type]
    results = service.search("тест", k=5, law_code="ГК РФ", on_date="2024-01-01")
    assert len(results) == 2
    main_filter = store.filters[0]
    assert any(cond.key == "law_code" for cond in main_filter.must)
    assert len(store.filters) >= 2
    cross_filter = store.filters[1]
    assert any(cond.key == "article" and cond.match.value == "395" for cond in cross_filter.must)
