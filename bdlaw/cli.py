"""Command line interface for BDlaw."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from bdlaw.app.pipeline import ApplicationPipelines
from bdlaw.gpt.answerer import AnswerChunk, Answerer
from bdlaw.gpt.validator import AnswerValidator
from bdlaw.parser.structure import ParsingContext
from bdlaw.search.service import SearchService
from bdlaw.settings.config import load_config
from bdlaw.testsuite.runner import TestSuiteRunner
from bdlaw.utils.logging import configure_logging


def build_context(args: argparse.Namespace) -> ParsingContext:
    return ParsingContext(
        jurisdiction=args.jurisdiction,
        act_type=args.act_type,
        law_code=args.law_code,
        law_full_title=args.law_full_title,
        version_id=args.version_id,
        valid_from=args.valid_from,
        valid_to=args.valid_to,
        supersedes=args.supersedes,
        source_url=args.source_url,
    )


def command_ingest(args: argparse.Namespace) -> None:
    config = load_config()
    configure_logging()
    pipelines = ApplicationPipelines(config)
    context = build_context(args)

    def log_progress(message: str) -> None:
        print(message)

    result = pipelines.ingest_paths([Path(p) for p in args.paths], context, progress=log_progress)
    stats = pipelines.index_parsed(result.parsed, progress=log_progress)
    print(f"Обработано документов: {len(result.documents)}")
    print(f"Индексировано норм: {stats.norms_indexed}")


def command_search(args: argparse.Namespace) -> None:
    config = load_config()
    pipelines = ApplicationPipelines(config)
    search = SearchService(pipelines.index_pipeline.embedder, pipelines.index_pipeline.store)
    results = search.search(args.query, k=args.k, law_code=args.law_code, on_date=args.date)
    for result in results:
        print(f"{result.citation} — score={result.score:.3f}")


def command_answer(args: argparse.Namespace) -> None:
    config = load_config()
    pipelines = ApplicationPipelines(config)
    search = SearchService(pipelines.index_pipeline.embedder, pipelines.index_pipeline.store)
    results = search.search(args.query, k=args.k, law_code=args.law_code, on_date=args.date)
    chunks = [
        AnswerChunk(
            citation=result.citation,
            source_url=result.payload.get("source_url", ""),
            clean_text=result.payload.get("clean_text", ""),
        )
        for result in results
    ]
    answerer = Answerer(config.llm.answer_model)
    answer = answerer.answer(args.query, chunks)
    print(answer.summary)
    validator = AnswerValidator(config.llm.judge_model)
    validation = validator.validate(answer.summary, answer.citations, [c.citation for c in chunks])
    print(f"Validation: {validation.verdict} — {validation.explanation}")


def command_tests(args: argparse.Namespace) -> None:
    config = load_config()
    pipelines = ApplicationPipelines(config)
    search = SearchService(pipelines.index_pipeline.embedder, pipelines.index_pipeline.store)
    runner = TestSuiteRunner(search, k=args.k)
    suite = TestSuiteRunner.load_suite(Path(args.suite))
    report = runner.run(suite)
    print(f"Recall@{args.k}: {report.recall_at_k:.3f}")
    print(f"Precision@{args.k}: {report.precision_at_k:.3f}")
    print(f"MRR@{args.k}: {report.mrr_at_k:.3f}")


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="BDlaw CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Импорт и индексация документов")
    ingest.add_argument("paths", nargs="+", help="Файлы или директории")
    ingest.add_argument("--jurisdiction", default="ru")
    ingest.add_argument("--act-type", dest="act_type", default="federal_law")
    ingest.add_argument("--law-code", dest="law_code", required=True)
    ingest.add_argument("--law-full-title", dest="law_full_title", required=True)
    ingest.add_argument("--version-id", dest="version_id", required=True)
    ingest.add_argument("--valid-from", dest="valid_from", required=True)
    ingest.add_argument("--valid-to", dest="valid_to")
    ingest.add_argument("--supersedes")
    ingest.add_argument("--source-url", dest="source_url")
    ingest.set_defaults(func=command_ingest)

    search_parser = subparsers.add_parser("search", help="Семантический поиск")
    search_parser.add_argument("--query", required=True)
    search_parser.add_argument("--law-code", dest="law_code")
    search_parser.add_argument("--date")
    search_parser.add_argument("--k", type=int, default=5)
    search_parser.set_defaults(func=command_search)

    answer_parser = subparsers.add_parser("answer", help="Ответ с цитатами")
    answer_parser.add_argument("--query", required=True)
    answer_parser.add_argument("--law-code", dest="law_code")
    answer_parser.add_argument("--date")
    answer_parser.add_argument("--k", type=int, default=5)
    answer_parser.set_defaults(func=command_answer)

    tests_parser = subparsers.add_parser("tests", help="Запуск golden-set")
    tests_parser.add_argument("suite", help="Путь к JSON golden-set")
    tests_parser.add_argument("--k", type=int, default=5)
    tests_parser.set_defaults(func=command_tests)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
