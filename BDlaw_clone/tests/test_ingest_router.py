from types import SimpleNamespace
from pathlib import Path

from app.data_repository import DataRepository
from app.ingest_router import IngestionRouter, RouterSniffInfo
from app.settings import AppSettings


def test_router_cache_reuses_plan(monkeypatch, tmp_path):
    settings = AppSettings()
    settings.router.use_llm = False
    repository = DataRepository(tmp_path / "data")
    router = IngestionRouter(settings, repository)

    sniff = RouterSniffInfo(
        path=tmp_path / "doc.rtf",
        sample_text="Статья 1. Общие положения",
        headings=["Статья 1"],
        language="ru",
        mime_type="text/rtf",
    )

    plan_first = router.route(sniff)
    assert plan_first.doc_type == "legal_code"

    def _boom(sniff_info):
        raise AssertionError("router should have served cached plan")

    monkeypatch.setattr(router, "_generate_plan", _boom)
    plan_second = router.route(sniff)
    assert plan_second.model_dump() == plan_first.model_dump()


def test_router_falls_back_when_llm_errors(monkeypatch, tmp_path):
    settings = AppSettings()
    settings.router.use_llm = True
    settings.openai_api_key = "test"
    repository = DataRepository(tmp_path / "data")
    router = IngestionRouter(settings, repository)

    sniff = RouterSniffInfo(
        path=tmp_path / "doc.rtf",
        sample_text="Статья 2",
        headings=["Статья 2"],
        language="ru",
        mime_type="text/rtf",
    )

    monkeypatch.setattr(router, "_client", SimpleNamespace(responses=SimpleNamespace(create=lambda **_: (_ for _ in ()).throw(RuntimeError("boom")))))
    plan = router.route(sniff)
    assert plan.doc_type == "legal_code"
