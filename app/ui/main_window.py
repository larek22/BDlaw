from __future__ import annotations

import html
import logging
import re
import time
from pathlib import Path
from typing import Sequence, cast

from PySide6.QtCore import QObject, QThread, Signal, Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QPlainTextEdit,
    QSpinBox,
    QTabWidget,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..embeddings import EmbeddingClient
from ..ingest import IngestService
from ..logging_config import configure_logging
from ..query_pipeline import QueryPipeline
from ..qdrant_client import QdrantVectorStore
from ..settings import AppSettings
from ..vectorization.models import DistanceMetric, VectorizationPlan
from ..vectorization.service import VectorizationService
from .vectorization_plan_dialog import VectorizationPlanDialog

logger = logging.getLogger(__name__)


class IngestWorker(QObject):
    finished = Signal(dict)
    error = Signal(str)
    progress = Signal(str)

    def __init__(
        self,
        service: IngestService,
        paths: Sequence[Path],
        recreate: bool,
        approved_plans: dict[Path, VectorizationPlan] | None = None,
    ) -> None:
        super().__init__()
        self.service = service
        self.paths = paths
        self.recreate = recreate
        self.approved_plans = approved_plans or {}

    def run(self) -> None:  # pragma: no cover - requires Qt thread
        try:
            stats = self.service.ingest(
                self.paths,
                recreate=self.recreate,
                approved_plans=self.approved_plans,
                progress_cb=self.progress.emit,
            )
            self.finished.emit(
                {
                    "files_processed": stats.files_processed,
                    "chunks_created": stats.chunks_created,
                    "skipped": stats.skipped,
                }
            )
        except Exception as exc:  # pragma: no cover - runtime errors
            logger.exception("Ingestion failed")
            self.error.emit(str(exc))


class QueryWorker(QObject):
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, pipeline: QueryPipeline, question: str) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.question = question

    def run(self) -> None:  # pragma: no cover
        try:
            result = self.pipeline.answer(self.question)
            self.finished.emit(
                {
                    "answer": result.answer,
                    "sources": [
                        {
                            "doc_id": source.chunk.doc_id,
                            "chunk_index": source.chunk.chunk_index,
                            "title": source.chunk.title_text,
                            "body": source.chunk.body_text,
                            "hierarchy": source.chunk.hierarchy,
                            "score": source.score,
                        }
                        for source in result.sources
                    ],
                    "question": self.question,
                }
            )
        except Exception as exc:
            logger.exception("Query failed")
            self.error.emit(str(exc))


class SettingsTab(QWidget):
    settings_saved = Signal(AppSettings)

    def __init__(self, settings: AppSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.settings = settings
        self._build_ui()
        self._load_settings()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        form = QFormLayout()
        self.openai_key_input = QLineEdit()
        self.openai_key_input.setEchoMode(QLineEdit.Password)
        form.addRow("OpenAI API Key", self.openai_key_input)

        self.embedding_model_input = QLineEdit()
        form.addRow("Embedding Model", self.embedding_model_input)

        self.chat_model_input = QLineEdit()
        form.addRow("Chat Model", self.chat_model_input)

        self.qdrant_url_input = QLineEdit()
        form.addRow("Qdrant URL", self.qdrant_url_input)

        self.qdrant_key_input = QLineEdit()
        self.qdrant_key_input.setEchoMode(QLineEdit.Password)
        form.addRow("Qdrant API Key", self.qdrant_key_input)

        self.collection_input = QLineEdit()
        form.addRow("Collection", self.collection_input)

        self.chunk_size_input = QSpinBox()
        self.chunk_size_input.setRange(200, 5000)
        self.chunk_size_input.setSingleStep(100)
        form.addRow("Chunk Size (chars)", self.chunk_size_input)

        self.chunk_overlap_input = QSpinBox()
        self.chunk_overlap_input.setRange(0, 1000)
        self.chunk_overlap_input.setSingleStep(20)
        form.addRow("Chunk Overlap", self.chunk_overlap_input)

        self.top_k_input = QSpinBox()
        self.top_k_input.setRange(1, 20)
        form.addRow("Top-K", self.top_k_input)

        layout.addLayout(form)

        button_row = QHBoxLayout()
        self.save_button = QPushButton("Save Settings")
        self.save_button.clicked.connect(self.save_settings)
        button_row.addWidget(self.save_button)

        self.test_openai_button = QPushButton("Test OpenAI")
        button_row.addWidget(self.test_openai_button)

        self.test_qdrant_button = QPushButton("Test Qdrant")
        button_row.addWidget(self.test_qdrant_button)

        button_row.addStretch(1)

        layout.addLayout(button_row)
        layout.addStretch(1)

    def _load_settings(self) -> None:
        self.openai_key_input.setText(self.settings.openai_api_key)
        self.embedding_model_input.setText(self.settings.openai_models.embedding)
        self.chat_model_input.setText(self.settings.openai_models.chat)
        self.qdrant_url_input.setText(self.settings.qdrant.url)
        self.qdrant_key_input.setText(self.settings.qdrant.api_key)
        self.collection_input.setText(self.settings.qdrant.collection)
        self.chunk_size_input.setValue(self.settings.ingest.chunk_size_chars)
        self.chunk_overlap_input.setValue(self.settings.ingest.chunk_overlap_chars)
        self.top_k_input.setValue(self.settings.query.top_k)

    def save_settings(self) -> None:
        self.settings.openai_api_key = self.openai_key_input.text().strip()
        self.settings.openai_models.embedding = self.embedding_model_input.text().strip()
        self.settings.openai_models.chat = self.chat_model_input.text().strip()
        self.settings.qdrant.url = self.qdrant_url_input.text().strip()
        self.settings.qdrant.api_key = self.qdrant_key_input.text().strip()
        self.settings.qdrant.collection = self.collection_input.text().strip()
        self.settings.ingest.chunk_size_chars = self.chunk_size_input.value()
        self.settings.ingest.chunk_overlap_chars = self.chunk_overlap_input.value()
        self.settings.query.top_k = self.top_k_input.value()
        self.settings.save()
        self.settings_saved.emit(self.settings)
        QMessageBox.information(self, "Settings", "Settings saved successfully")


class IngestTab(QWidget):
    start_ingest = Signal(list, bool)
    reset_requested = Signal()
    plan_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        button_row = QHBoxLayout()
        self.add_files_button = QPushButton("Add Files...")
        self.add_files_button.clicked.connect(self._open_file_dialog)
        button_row.addWidget(self.add_files_button)

        self.plan_button = QPushButton("Analyze (GPT Plan)")
        self.plan_button.clicked.connect(self._emit_plan)
        self.plan_button.setEnabled(False)
        button_row.addWidget(self.plan_button)

        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_files)
        button_row.addWidget(self.clear_button)

        self.reset_button = QPushButton("Reset DB")
        self.reset_button.clicked.connect(self._emit_reset)
        button_row.addWidget(self.reset_button)

        self.rebuild_checkbox = QCheckBox("Rebuild Collection")
        button_row.addWidget(self.rebuild_checkbox)

        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.file_list = QListWidget()
        layout.addWidget(self.file_list)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        self.log_output = QPlainTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setMaximumBlockCount(1000)
        layout.addWidget(self.log_output)

        self.ingest_button = QPushButton("Create Vector DB")
        self.ingest_button.clicked.connect(self._emit_ingest)
        self.ingest_button.setEnabled(False)
        layout.addWidget(self.ingest_button)

    def _open_file_dialog(self) -> None:
        dialog = QFileDialog(self)
        dialog.setFileMode(QFileDialog.ExistingFiles)
        dialog.setNameFilters([
            "Documents (*.pdf *.docx *.txt *.rtf)",
            "PDF (*.pdf)",
            "Word (*.docx)",
            "Text (*.txt)",
            "Rich Text (*.rtf)",
        ])
        if dialog.exec():
            files = dialog.selectedFiles()
            for file_path in files:
                self.file_list.addItem(file_path)
        self._update_button_state()

    def _update_button_state(self) -> None:
        self.ingest_button.setEnabled(self.file_list.count() > 0)
        self.plan_button.setEnabled(self.file_list.count() > 0)

    def clear_files(self) -> None:
        self.file_list.clear()
        self._update_button_state()

    def _emit_ingest(self) -> None:
        paths = [self.file_list.item(i).text() for i in range(self.file_list.count())]
        recreate = self.rebuild_checkbox.isChecked()
        self.start_ingest.emit(paths, recreate)

    def set_running(self, running: bool) -> None:
        self.add_files_button.setEnabled(not running)
        self.plan_button.setEnabled((not running) and self.file_list.count() > 0)
        self.clear_button.setEnabled(not running)
        self.reset_button.setEnabled(not running)
        self.ingest_button.setEnabled((not running) and self.file_list.count() > 0)
        if running:
            self.progress_bar.show()
        else:
            self.progress_bar.hide()

    def append_log(self, message: str) -> None:
        self.log_output.appendPlainText(message)

    def _emit_reset(self) -> None:
        self.reset_requested.emit()

    def _emit_plan(self) -> None:
        if self.file_list.count() == 0:
            QMessageBox.warning(self, "Analyze", "Please add a file first")
            return
        item = self.file_list.currentItem()
        if item is None:
            item = self.file_list.item(0)
        if item is None:
            QMessageBox.warning(self, "Analyze", "No file selected")
            return
        self.plan_requested.emit(item.text())


class QueryTab(QWidget):
    start_query = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        self.question_input = QPlainTextEdit()
        self.question_input.setPlaceholderText("Ask a question about the ingested documents...")
        layout.addWidget(self.question_input)

        self.ask_button = QPushButton("Search & Answer")
        self.ask_button.clicked.connect(self._on_ask)
        layout.addWidget(self.ask_button)

        self.answer_output = QTextEdit()
        self.answer_output.setReadOnly(True)
        layout.addWidget(self.answer_output)

        self.sources_browser = QTextBrowser()
        self.sources_browser.setOpenExternalLinks(True)
        layout.addWidget(self.sources_browser)

    def _on_ask(self) -> None:
        question = self.question_input.toPlainText().strip()
        if not question:
            QMessageBox.warning(self, "Ask", "Please enter a question")
            return
        self.start_query.emit(question)

    def set_running(self, running: bool) -> None:
        self.ask_button.setEnabled(not running)

    def display_answer(self, answer: str, sources: Sequence[dict], query: str) -> None:
        self.answer_output.setPlainText(answer)
        self.sources_browser.clear()
        terms = set(word.lower() for word in re.findall(r"\w+", query))
        html_parts = []
        for idx, source in enumerate(sources, start=1):
            body_text = source.get("body", "")
            highlighted = self._highlight_terms(body_text, terms)
            hierarchy = source.get("hierarchy", {}) or {}
            trail_parts = []
            if hierarchy.get("part_no"):
                trail_parts.append(f"Part {hierarchy['part_no']}")
            if hierarchy.get("article_no"):
                trail_parts.append(f"Article {hierarchy['article_no']}")
            location = " · ".join(trail_parts) if trail_parts else f"chunk {source.get('chunk_index')}"
            doc_id = html.escape(source.get("doc_id", ""))
            title = html.escape(source.get("title", ""))
            header = title or location
            html_parts.append(
                f"<p><b>[{idx}] {doc_id} — {header}</b><br>{highlighted}</p>"
            )
        self.sources_browser.setHtml("".join(html_parts))

    def _highlight_terms(self, text: str, terms: set[str]) -> str:
        escaped = html.escape(text)
        if not terms:
            return escaped
        pattern = re.compile(r"(" + "|".join(re.escape(term) for term in terms if term) + r")", re.IGNORECASE)

        def replacer(match: re.Match[str]) -> str:
            return f"<span style='background-color: #fff2a8'>{match.group(0)}</span>"

        return pattern.sub(replacer, escaped)


class MainWindow(QMainWindow):
    def __init__(self, settings: AppSettings) -> None:
        super().__init__()
        configure_logging()
        self.settings = settings
        self.embedding_client: EmbeddingClient | None = None
        self.vector_store: QdrantVectorStore | None = None
        self.ingest_service: IngestService | None = None
        self.query_pipeline: QueryPipeline | None = None
        self.vectorization_service: VectorizationService | None = None
        self._approved_plans: dict[Path, VectorizationPlan] = {}
        self._ingest_thread: QThread | None = None
        self._ingest_worker: IngestWorker | None = None
        self._query_thread: QThread | None = None
        self._query_worker: QueryWorker | None = None

        self.setWindowTitle("Vector KB Assistant")
        self.resize(900, 700)

        self._build_ui()
        self._init_services()

    def _build_ui(self) -> None:
        self.tabs = QTabWidget()
        self.settings_tab = SettingsTab(self.settings)
        self.ingest_tab = IngestTab()
        self.query_tab = QueryTab()

        self.tabs.addTab(self.settings_tab, "Settings")
        self.tabs.addTab(self.ingest_tab, "Ingest")
        self.tabs.addTab(self.query_tab, "Ask")
        self.setCentralWidget(self.tabs)

        self.settings_tab.settings_saved.connect(self._init_services)
        self.settings_tab.test_openai_button.clicked.connect(self._test_openai)
        self.settings_tab.test_qdrant_button.clicked.connect(self._test_qdrant)

        self.ingest_tab.start_ingest.connect(self._start_ingest)
        self.ingest_tab.reset_requested.connect(self._reset_collection)
        self.ingest_tab.plan_requested.connect(self._analyze_plan)
        self.query_tab.start_query.connect(self._start_query)

    def _init_services(self) -> None:
        try:
            self.embedding_client = EmbeddingClient(self.settings)
            self.vector_store = QdrantVectorStore(self.settings)
            self.ingest_service = IngestService(self.settings, self.embedding_client, self.vector_store)
            self.query_pipeline = QueryPipeline(self.settings, self.embedding_client, self.vector_store)
            self.vectorization_service = VectorizationService()
        except Exception as exc:
            QMessageBox.critical(self, "Initialization error", str(exc))
            logger.exception("Failed to initialize services")
            self.vectorization_service = None

    def _test_openai(self) -> None:
        try:
            client = EmbeddingClient(self.settings)
            from hashlib import sha256

            text = "ping"
            sha = sha256(text.encode("utf-8")).hexdigest()
            began = time.perf_counter()
            result = client.embed_texts([text], [sha])[0]
            elapsed = time.perf_counter() - began
            dimension = len(result.vector)
            QMessageBox.information(
                self,
                "OpenAI",
                f"OpenAI connection successful\nDimension: {dimension}\nLatency: {elapsed:.2f}s",
            )
        except Exception as exc:
            QMessageBox.critical(self, "OpenAI", f"OpenAI test failed: {exc}")

    def _test_qdrant(self) -> None:
        try:
            store = QdrantVectorStore(self.settings)
            client = store.client
            try:
                collections_response = client.get_collections()
                collections = getattr(collections_response, "collections", []) or []
            except Exception as exc:
                QMessageBox.critical(self, "Qdrant", f"Failed to list collections: {exc}")
                return

            alias_target = "<unset>"
            try:
                aliases = client.get_aliases()
                for description in getattr(aliases, "aliases", []) or []:
                    if getattr(description, "alias_name", None) == store.alias_name:
                        alias_target = getattr(description, "collection_name", "<unknown>")
                        break
            except Exception as exc:
                alias_target = f"alias lookup failed: {exc}"

            message = (
                f"Endpoint: {store.endpoint_url}\n"
                f"Server version: {store.server_version or '<unknown>'}\n"
                f"Collections discovered: {len(collections)}\n"
                f"Alias '{store.alias_name}' → {alias_target}"
            )
            QMessageBox.information(self, "Qdrant", message)
        except Exception as exc:
            QMessageBox.critical(self, "Qdrant", f"Qdrant test failed: {exc}")

    def _reset_collection(self) -> None:
        if not self.vector_store:
            QMessageBox.critical(self, "Reset DB", "Vector store is not initialised")
            return
        confirm = QMessageBox.question(
            self,
            "Reset Vector DB",
            "This will drop and recreate the configured Qdrant collection. Continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        try:
            embedding_model = self.settings.openai_models.embedding
            self.vector_store.reset_collection(embedding_model)
        except Exception as exc:
            logger.exception("Failed to reset Qdrant collection")
            message = f"Failed to reset collection: {exc}"
            if self.ingest_tab:
                self.ingest_tab.append_log(message)
            QMessageBox.critical(self, "Reset DB", message)
        else:
            message = "Collection reset successfully."
            if self.ingest_tab:
                self.ingest_tab.append_log(message)
            QMessageBox.information(self, "Reset DB", message)

    def _analyze_plan(self, path: str) -> None:
        path_obj = Path(path).resolve()
        if not path_obj.exists():
            QMessageBox.critical(self, "Analyze", f"File not found: {path_obj}")
            return
        if self.vectorization_service is None:
            self.vectorization_service = VectorizationService()
        service = self.vectorization_service
        distance_raw = getattr(self.settings.qdrant, "distance", "cosine")
        distance: DistanceMetric
        if distance_raw in ("cosine", "dot", "euclid"):
            distance = cast(DistanceMetric, distance_raw)
        else:
            distance = cast(DistanceMetric, "cosine")
        chat_model = self.settings.openai_models.chat if self.settings.openai_api_key else None
        try:
            artifacts = service.analyze(
                path=path_obj,
                collection_name=self.settings.qdrant.collection,
                embedding_model=self.settings.openai_models.embedding,
                distance=distance,
                hard_cap=self.settings.chunking.max_tokens,
                chat_model=chat_model,
                api_key=self.settings.openai_api_key,
            )
        except Exception as exc:
            logger.exception("Vectorization plan analysis failed")
            QMessageBox.critical(self, "Analyze", f"Plan generation failed: {exc}")
            return
        dialog = VectorizationPlanDialog(artifacts.preview, self)
        result = dialog.exec()
        if result == QDialog.Accepted:
            selected_plan = dialog.plan()
            self._approved_plans[path_obj] = selected_plan
            self.ingest_tab.append_log(f"Plan approved for {path_obj.name}")
        else:
            self.ingest_tab.append_log(f"Plan cancelled for {path_obj.name}")

    def _start_ingest(self, paths: Sequence[str], recreate: bool) -> None:
        if not paths:
            return
        if not self.ingest_service:
            QMessageBox.critical(self, "Ingest", "Services not initialized")
            return
        file_paths = [Path(p).resolve() for p in paths]
        for path_obj in file_paths:
            if path_obj in self._approved_plans:
                self.ingest_tab.append_log(f"Using approved plan for {path_obj.name}")
            else:
                self.ingest_tab.append_log(f"No approved plan for {path_obj.name}; fallback will be used")
        self.ingest_tab.set_running(True)
        self.ingest_tab.append_log("Starting ingestion...")
        self._cleanup_worker("_ingest_thread", "_ingest_worker")
        worker = IngestWorker(
            self.ingest_service,
            file_paths,
            recreate,
            approved_plans=self._approved_plans.copy(),
        )
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._ingest_finished, Qt.QueuedConnection)
        worker.error.connect(self._ingest_error, Qt.QueuedConnection)
        worker.progress.connect(self.ingest_tab.append_log, Qt.QueuedConnection)
        self._ingest_thread = thread
        self._ingest_worker = worker
        thread.start()

    def _ingest_finished(self, result: dict) -> None:
        self.ingest_tab.append_log(
            f"Ingestion complete: {result['files_processed']} files, {result['chunks_created']} chunks, {result['skipped']} skipped"
        )
        self.ingest_tab.set_running(False)
        self._cleanup_worker("_ingest_thread", "_ingest_worker")

    def _ingest_error(self, error: str) -> None:
        self.ingest_tab.append_log(f"Error: {error}")
        QMessageBox.critical(self, "Ingest", error)
        self.ingest_tab.set_running(False)
        self._cleanup_worker("_ingest_thread", "_ingest_worker")

    def _start_query(self, question: str) -> None:
        if not self.query_pipeline:
            QMessageBox.critical(self, "Ask", "Services not initialized")
            return
        self.query_tab.set_running(True)
        self._cleanup_worker("_query_thread", "_query_worker")
        worker = QueryWorker(self.query_pipeline, question)
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._query_finished, Qt.QueuedConnection)
        worker.error.connect(self._query_error, Qt.QueuedConnection)
        self._query_thread = thread
        self._query_worker = worker
        thread.start()

    def _query_finished(self, result: dict) -> None:
        question = result.get("question", "")
        self.query_tab.display_answer(result.get("answer", ""), result.get("sources", []), question)
        self.query_tab.set_running(False)
        self._cleanup_worker("_query_thread", "_query_worker")

    def _query_error(self, error: str) -> None:
        QMessageBox.critical(self, "Ask", error)
        self.query_tab.set_running(False)
        self._cleanup_worker("_query_thread", "_query_worker")

    def _cleanup_worker(self, thread_attr: str, worker_attr: str) -> None:
        thread = getattr(self, thread_attr)
        worker = getattr(self, worker_attr)
        if thread:
            current = QThread.currentThread()
            thread.quit()
            if thread is not current:
                thread.wait()
            thread.deleteLater()
        if worker:
            worker.deleteLater()
        setattr(self, thread_attr, None)
        setattr(self, worker_attr, None)


def run_app() -> None:  # pragma: no cover
    import sys

    configure_logging()
    settings = AppSettings.load()
    app = QApplication(sys.argv)
    window = MainWindow(settings)
    window.show()
    sys.exit(app.exec())
