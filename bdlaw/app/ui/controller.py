"""Application controller that bridges GUI events with pipelines."""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Iterable, List, Optional

from PySide6 import QtCore

from bdlaw.app.pipeline import ApplicationPipelines
from bdlaw.gpt.answerer import Answer, AnswerChunk, Answerer
from bdlaw.gpt.validator import AnswerValidator, ValidationResult
from bdlaw.parser.structure import ParsingContext
from bdlaw.search.service import SearchResult
from bdlaw.settings.config import AppConfig


class IngestThread(QtCore.QThread):
    """Background worker performing ingestion and indexing."""

    progress = QtCore.Signal(str)
    completed = QtCore.Signal(int, int)
    failed = QtCore.Signal(str)

    def __init__(
        self,
        pipelines: ApplicationPipelines,
        paths: Iterable[Path],
        context: ParsingContext,
    ) -> None:
        super().__init__()
        self._pipelines = pipelines
        self._paths = list(paths)
        self._context = context

    def run(self) -> None:  # pragma: no cover - threaded execution
        try:
            result = self._pipelines.ingest_paths(self._paths, self._context, progress=self.progress.emit)
            stats = self._pipelines.index_parsed(result.parsed, progress=self.progress.emit)
            self.completed.emit(len(result.documents), stats.norms_indexed)
        except Exception as exc:  # noqa: BLE001 - show details to the user
            self.failed.emit("\n".join([str(exc), traceback.format_exc()]))


class SearchThread(QtCore.QThread):
    """Background search worker to keep UI responsive."""

    completed = QtCore.Signal(list)
    failed = QtCore.Signal(str)

    def __init__(self, controller: "AppController", query: str, k: int, law_code: Optional[str], on_date: Optional[str]) -> None:
        super().__init__()
        self._controller = controller
        self._query = query
        self._k = k
        self._law_code = law_code
        self._on_date = on_date

    def run(self) -> None:  # pragma: no cover - threaded execution
        try:
            results = self._controller._search_service.search(
                self._query,
                k=self._k,
                law_code=self._law_code,
                on_date=self._on_date,
            )
            self.completed.emit(results)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class AnswerThread(QtCore.QThread):
    """Background GPT answer worker."""

    completed = QtCore.Signal(Answer, ValidationResult)
    failed = QtCore.Signal(str)

    def __init__(self, controller: "AppController", question: str, chunks: List[AnswerChunk]) -> None:
        super().__init__()
        self._controller = controller
        self._question = question
        self._chunks = chunks

    def run(self) -> None:  # pragma: no cover - threaded execution
        try:
            answer = self._controller._answerer.answer(self._question, self._chunks)
            validation = self._controller._validator.validate(
                answer.summary,
                answer.citations,
                [chunk.citation for chunk in self._chunks],
            )
            self.completed.emit(answer, validation)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class AppController(QtCore.QObject):
    """Main glue layer between UI and backend services."""

    log_message = QtCore.Signal(str)
    ingest_started = QtCore.Signal(int)
    ingest_finished = QtCore.Signal(int, int)
    ingest_failed = QtCore.Signal(str)
    search_finished = QtCore.Signal(list)
    search_failed = QtCore.Signal(str)
    answer_finished = QtCore.Signal(Answer, ValidationResult)
    answer_failed = QtCore.Signal(str)

    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self._pipelines = ApplicationPipelines(config)
        self._search_service = self._pipelines.build_search_service()
        self._answerer = Answerer(config.llm.answer_model)
        self._validator = AnswerValidator(config.llm.judge_model)
        self._ingest_thread: Optional[IngestThread] = None
        self._search_threads: List[SearchThread] = []
        self._answer_threads: List[AnswerThread] = []

    # ------------------------------------------------------------------
    # Ingestion / indexing
    # ------------------------------------------------------------------
    def ingest_and_index(self, paths: Iterable[Path], context: ParsingContext) -> None:
        """Kick off background ingestion for *paths*."""

        if self._ingest_thread and self._ingest_thread.isRunning():
            self.log_message.emit("Идёт индексация — дождитесь завершения.")
            return

        paths = list(paths)
        if not paths:
            return
        self.ingest_started.emit(len(paths))
        thread = IngestThread(self._pipelines, paths, context)
        thread.progress.connect(self.log_message.emit)
        thread.completed.connect(self.ingest_finished.emit)
        thread.failed.connect(self.ingest_failed.emit)
        thread.completed.connect(self._clear_ingest_thread)
        thread.failed.connect(self._clear_ingest_thread)
        self._ingest_thread = thread
        thread.start()

    def _clear_ingest_thread(self, *args: object) -> None:
        if self._ingest_thread:
            self._ingest_thread.deleteLater()
            self._ingest_thread = None

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def perform_search(self, query: str, k: int = 5, law_code: Optional[str] = None, on_date: Optional[str] = None) -> None:
        if not query.strip():
            return
        thread = SearchThread(self, query, k, law_code, on_date)
        thread.completed.connect(self.search_finished.emit)
        thread.failed.connect(self.search_failed.emit)
        thread.finished.connect(lambda: self._cleanup_thread(thread, self._search_threads))
        thread.failed.connect(lambda _msg: self._cleanup_thread(thread, self._search_threads))
        self._search_threads.append(thread)
        thread.start()

    # ------------------------------------------------------------------
    # Answers
    # ------------------------------------------------------------------
    def generate_answer(self, question: str, search_results: List[SearchResult]) -> None:
        if not search_results:
            self.answer_failed.emit("Нет результатов для формирования ответа")
            return
        chunks = [
            AnswerChunk(
                citation=result.citation,
                source_url=result.payload.get("source_url", ""),
                clean_text=result.payload.get("clean_text", ""),
            )
            for result in search_results
        ]
        thread = AnswerThread(self, question, chunks)
        thread.completed.connect(self.answer_finished.emit)
        thread.failed.connect(self.answer_failed.emit)
        thread.finished.connect(lambda: self._cleanup_thread(thread, self._answer_threads))
        thread.failed.connect(lambda _msg: self._cleanup_thread(thread, self._answer_threads))
        self._answer_threads.append(thread)
        thread.start()

    def _cleanup_thread(self, thread: QtCore.QThread, container: List[QtCore.QThread]) -> None:
        if thread in container:
            container.remove(thread)
        thread.deleteLater()


__all__ = ["AppController"]
