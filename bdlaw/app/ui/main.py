"""Main PySide6 window for the BDlaw desktop toolkit."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Callable, Dict, List

from PySide6 import QtCore, QtGui, QtWidgets

from bdlaw.app.ui.controller import AppController
from bdlaw.app.ui.dialogs import MetadataDialog
from bdlaw.parser.structure import ParsingContext
from bdlaw.search.service import SearchResult
from bdlaw.settings.config import AppConfig


class MainWindow(QtWidgets.QMainWindow):
    """Primary application window orchestrating user flows."""

    def __init__(self, config: AppConfig, controller: AppController):
        super().__init__()
        self.config = config
        self.controller = controller
        self.setWindowTitle("BDlaw — юридический RAG-конструктор")
        self.resize(1280, 860)
        self.statusBar().showMessage("Готово к работе")

        self._metadata_defaults: Dict[str, str] = {
            "jurisdiction": "ru",
            "act_type": "federal_law",
        }
        self._last_search_results: List[SearchResult] = []

        self._setup_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------
    def _setup_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(central)

        toolbar = QtWidgets.QToolBar()
        toolbar.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        toolbar.addAction(self._create_action("Импорт", self._show_import, "Ctrl+I"))
        toolbar.addAction(self._create_action("Индексировать", self._show_index, "Ctrl+Enter"))
        toolbar.addAction(self._create_action("Поиск", self._show_search, "Ctrl+F"))
        toolbar.addAction(self._create_action("Тесты", self._show_tests, "F5"))
        toolbar.addAction(self._create_action("Настройки", self._show_settings, "Ctrl+Comma"))
        self.addToolBar(toolbar)

        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self._build_home())
        self.stack.addWidget(self._build_import())
        self.stack.addWidget(self._build_index())
        self.stack.addWidget(self._build_search())
        self.stack.addWidget(self._build_tests())

        layout.addWidget(self.stack)
        self.setCentralWidget(central)

    def _connect_signals(self) -> None:
        self.controller.log_message.connect(self._append_log)
        self.controller.ingest_started.connect(self._on_ingest_started)
        self.controller.ingest_finished.connect(self._on_ingest_finished)
        self.controller.ingest_failed.connect(self._on_ingest_failed)
        self.controller.search_finished.connect(self._on_search_finished)
        self.controller.search_failed.connect(self._on_search_failed)
        self.controller.answer_finished.connect(self._on_answer_finished)
        self.controller.answer_failed.connect(self._on_answer_failed)

    def _create_action(self, text: str, callback: Callable[[], None], shortcut: str) -> QtGui.QAction:
        action = QtGui.QAction(text, self)
        action.triggered.connect(callback)
        action.setShortcut(QtGui.QKeySequence(shortcut))
        return action

    # ------------------------------------------------------------------
    # Page builders
    # ------------------------------------------------------------------
    def _build_home(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.addStretch()
        for label, callback in [
            ("Импорт и индексация", self._show_import),
            ("Поиск", self._show_search),
            ("Тесты", self._show_tests),
            ("Настройки", self._show_settings),
        ]:
            button = QtWidgets.QPushButton(label)
            button.setFixedHeight(70)
            button.clicked.connect(callback)
            layout.addWidget(button)
        layout.addStretch()
        return widget

    def _build_import(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        info = QtWidgets.QLabel("Выберите файлы или перетащите их в окно для подготовки индекса")
        info.setWordWrap(True)
        layout.addWidget(info)
        self.import_list = QtWidgets.QListWidget()
        self.import_list.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        layout.addWidget(self.import_list)
        layout.addStretch()
        return widget

    def _build_index(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        self.index_progress = QtWidgets.QProgressBar()
        self.index_progress.setRange(0, 1)
        self.index_progress.setValue(0)
        layout.addWidget(self.index_progress)
        self.index_log = QtWidgets.QPlainTextEdit()
        self.index_log.setReadOnly(True)
        layout.addWidget(self.index_log)
        return widget

    def _build_search(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        form = QtWidgets.QFormLayout()
        self.search_query = QtWidgets.QLineEdit()
        self.search_query.setPlaceholderText("Введите вопрос или ключевую фразу")
        form.addRow("Запрос", self.search_query)

        self.search_law = QtWidgets.QLineEdit()
        self.search_law.setPlaceholderText("Например: ГК РФ")
        form.addRow("Закон", self.search_law)

        self.search_date = QtWidgets.QLineEdit()
        self.search_date.setPlaceholderText("Дата на момент действия (ГГГГ-ММ-ДД)")
        form.addRow("На дату", self.search_date)

        self.search_k = QtWidgets.QSpinBox()
        self.search_k.setRange(1, 50)
        self.search_k.setValue(5)
        form.addRow("Top-K", self.search_k)

        layout.addLayout(form)

        buttons = QtWidgets.QHBoxLayout()
        self.search_button = QtWidgets.QPushButton("Искать")
        self.search_button.clicked.connect(self._perform_search)
        buttons.addWidget(self.search_button)
        self.answer_button = QtWidgets.QPushButton("Сформировать ответ (GPT)")
        self.answer_button.clicked.connect(self._request_answer)
        self.answer_button.setEnabled(False)
        buttons.addWidget(self.answer_button)
        layout.addLayout(buttons)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.search_results = QtWidgets.QListWidget()
        self.search_results.itemSelectionChanged.connect(self._update_selected_result)
        splitter.addWidget(self.search_results)

        right_panel = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(QtWidgets.QLabel("Текст нормы"))
        self.search_details = QtWidgets.QPlainTextEdit()
        self.search_details.setReadOnly(True)
        right_layout.addWidget(self.search_details)
        right_layout.addWidget(QtWidgets.QLabel("Ответ GPT"))
        self.answer_output = QtWidgets.QPlainTextEdit()
        self.answer_output.setReadOnly(True)
        right_layout.addWidget(self.answer_output)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

        layout.addWidget(splitter)
        return widget

    def _build_tests(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        self.tests_log = QtWidgets.QPlainTextEdit()
        self.tests_log.setReadOnly(True)
        layout.addWidget(self.tests_log)
        layout.addStretch()
        return widget

    # ------------------------------------------------------------------
    # Navigation handlers
    # ------------------------------------------------------------------
    def _show_import(self) -> None:
        self.stack.setCurrentIndex(1)
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Выберите документы", str(Path.home()))
        if not files:
            return
        dialog = MetadataDialog(self, defaults=self._metadata_defaults)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        context = dialog.get_context()
        self._remember_context(context)
        self.import_list.clear()
        self.import_list.addItems(files)
        self.stack.setCurrentIndex(2)
        self.controller.ingest_and_index([Path(f) for f in files], context)

    def _show_index(self) -> None:
        self.stack.setCurrentIndex(2)

    def _show_search(self) -> None:
        self.stack.setCurrentIndex(3)

    def _show_tests(self) -> None:
        self.stack.setCurrentIndex(4)

    def _show_settings(self) -> None:
        QtWidgets.QMessageBox.information(
            self,
            "Настройки",
            "Измените config.yaml и используйте системный keyring для ключей API.",
        )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------
    def _on_ingest_started(self, count: int) -> None:
        self.index_progress.setRange(0, 0)  # busy indicator
        self.statusBar().showMessage(f"Обработка файлов: {count}")
        self.index_log.clear()

    def _on_ingest_finished(self, documents: int, norms: int) -> None:
        self.index_progress.setRange(0, 1)
        self.index_progress.setValue(1)
        self.statusBar().showMessage(f"Индексирование завершено: {documents} файлов, {norms} норм")
        QtWidgets.QMessageBox.information(
            self,
            "Готово",
            f"Индексирование завершено. Обработано документов: {documents}. Норм: {norms}.",
        )

    def _on_ingest_failed(self, message: str) -> None:
        self.index_progress.setRange(0, 1)
        self.index_progress.setValue(0)
        self.statusBar().showMessage("Ошибка индексации")
        QtWidgets.QMessageBox.critical(self, "Ошибка", message)

    def _perform_search(self) -> None:
        query = self.search_query.text().strip()
        if not query:
            QtWidgets.QMessageBox.information(self, "Введите запрос", "Укажите текст запроса для поиска.")
            return
        law_code = self.search_law.text().strip() or None
        on_date = self.search_date.text().strip() or None
        k = self.search_k.value()
        self.search_button.setEnabled(False)
        self.statusBar().showMessage("Выполняется поиск…")
        self.controller.perform_search(query, k=k, law_code=law_code, on_date=on_date)

    def _on_search_finished(self, results: List[SearchResult]) -> None:
        self.search_button.setEnabled(True)
        self.statusBar().showMessage(f"Найдено результатов: {len(results)}")
        self._last_search_results = results
        self.search_results.clear()
        for result in results:
            item = QtWidgets.QListWidgetItem(f"{result.citation} — score={result.score:.3f}")
            self.search_results.addItem(item)
        self.answer_button.setEnabled(bool(results))
        self.answer_output.clear()
        if results:
            self.search_results.setCurrentRow(0)
        else:
            self.search_details.clear()

    def _on_search_failed(self, message: str) -> None:
        self.search_button.setEnabled(True)
        self.statusBar().showMessage("Ошибка поиска")
        QtWidgets.QMessageBox.critical(self, "Ошибка поиска", message)

    def _update_selected_result(self) -> None:
        index = self.search_results.currentRow()
        if index < 0 or index >= len(self._last_search_results):
            self.search_details.clear()
            return
        result = self._last_search_results[index]
        payload_text = result.payload.get("clean_text") or result.payload.get("raw_text") or ""
        meta = [
            f"Citation: {result.citation}",
            f"Score: {result.score:.3f}",
            f"Источник: {result.payload.get('source_url', '—')}",
        ]
        self.search_details.setPlainText("\n".join(meta) + "\n\n" + payload_text)

    def _request_answer(self) -> None:
        query = self.search_query.text().strip()
        if not query:
            QtWidgets.QMessageBox.information(self, "Введите запрос", "Введите вопрос для формирования ответа.")
            return
        if not self._last_search_results:
            QtWidgets.QMessageBox.information(self, "Нет данных", "Сначала выполните поиск и выберите результаты.")
            return
        self.answer_button.setEnabled(False)
        self.statusBar().showMessage("Формируется ответ GPT…")
        self.controller.generate_answer(query, self._last_search_results)

    def _on_answer_finished(self, answer, validation) -> None:
        self.answer_button.setEnabled(True)
        self.statusBar().showMessage(f"Ответ готов (валидация: {validation.verdict})")
        summary = answer.summary.strip()
        validation_text = f"Validation: {validation.verdict}\n{validation.explanation.strip()}"
        self.answer_output.setPlainText(f"{summary}\n\n{validation_text}")

    def _on_answer_failed(self, message: str) -> None:
        self.answer_button.setEnabled(True)
        self.statusBar().showMessage("Ошибка генерации ответа")
        QtWidgets.QMessageBox.critical(self, "GPT", message)

    def _append_log(self, message: str) -> None:
        self.index_log.appendPlainText(message)

    def _remember_context(self, context: ParsingContext) -> None:
        defaults: Dict[str, str] = {}
        for key, value in asdict(context).items():
            if isinstance(value, date):
                defaults[key] = value.isoformat()
            else:
                defaults[key] = str(value) if value is not None else ""
        self._metadata_defaults = defaults


def run_ui(config: AppConfig) -> None:
    app = QtWidgets.QApplication([])
    controller = AppController(config)
    window = MainWindow(config, controller)
    window.show()
    app.exec()


__all__ = ["MainWindow", "run_ui"]
