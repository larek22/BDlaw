"""PySide6 GUI entrypoint."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List

from PySide6 import QtCore, QtGui, QtWidgets

from bdlaw.settings.config import AppConfig


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, config: AppConfig, import_action: Callable[[List[Path]], None]):
        super().__init__()
        self.config = config
        self.import_action = import_action
        self.setWindowTitle("BDlaw")
        self.resize(1200, 800)
        self._setup_ui()

    def _setup_ui(self) -> None:
        central = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(central)

        toolbar = QtWidgets.QToolBar()
        toolbar.addAction(self._create_action("Импорт", self.import_action, "Ctrl+I"))
        toolbar.addAction(self._create_action("Поиск", self._show_search, "Ctrl+F"))
        toolbar.addAction(self._create_action("Индексировать", self._show_index, "Ctrl+Enter"))
        toolbar.addAction(self._create_action("Тесты", self._show_tests, "F5"))
        self.addToolBar(toolbar)

        self.stack = QtWidgets.QStackedWidget()
        self.stack.addWidget(self._build_home())
        self.stack.addWidget(self._build_import())
        self.stack.addWidget(self._build_index())
        self.stack.addWidget(self._build_search())
        self.stack.addWidget(self._build_tests())

        layout.addWidget(self.stack)
        self.setCentralWidget(central)

    def _create_action(self, text: str, callback: Callable[[], None], shortcut: str) -> QtGui.QAction:
        action = QtGui.QAction(text, self)
        action.triggered.connect(callback)
        action.setShortcut(QtGui.QKeySequence(shortcut))
        return action

    def _build_home(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.addStretch()
        for label, callback in [
            ("Импорт", self._show_import),
            ("Индексировать", self._show_index),
            ("Поиск", self._show_search),
            ("Тесты", self._show_tests),
            ("Настройки", self._show_settings),
        ]:
            button = QtWidgets.QPushButton(label)
            button.setFixedHeight(60)
            button.clicked.connect(callback)
            layout.addWidget(button)
        layout.addStretch()
        return widget

    def _build_import(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.addWidget(QtWidgets.QLabel("Drag & Drop файлы для импорта"))
        self.import_list = QtWidgets.QListWidget()
        layout.addWidget(self.import_list)
        return widget

    def _build_index(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        self.index_progress = QtWidgets.QProgressBar()
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
        form.addRow("Запрос", self.search_query)
        self.search_results = QtWidgets.QListWidget()
        layout.addLayout(form)
        layout.addWidget(self.search_results)
        return widget

    def _build_tests(self) -> QtWidgets.QWidget:
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        self.tests_log = QtWidgets.QPlainTextEdit()
        self.tests_log.setReadOnly(True)
        layout.addWidget(self.tests_log)
        return widget

    def _show_import(self) -> None:
        self.stack.setCurrentIndex(1)
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Выберите документы")
        if files:
            self.import_action([Path(f) for f in files])
            self.import_list.addItems(files)

    def _show_index(self) -> None:
        self.stack.setCurrentIndex(2)

    def _show_search(self) -> None:
        self.stack.setCurrentIndex(3)

    def _show_tests(self) -> None:
        self.stack.setCurrentIndex(4)

    def _show_settings(self) -> None:
        QtWidgets.QMessageBox.information(self, "Настройки", "Откройте config.yaml для изменения параметров")


def run_ui(config: AppConfig, import_action: Callable[[List[Path]], None]) -> None:
    app = QtWidgets.QApplication([])
    window = MainWindow(config, import_action)
    window.show()
    app.exec()


__all__ = ["MainWindow", "run_ui"]
