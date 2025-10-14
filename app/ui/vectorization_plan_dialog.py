"""Qt dialog that previews vectorization plans."""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QDialog,
    QDialogButtonBox,
    QMessageBox,
    QPlainTextEdit,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..vectorization.models import PlanPreview, VectorizationPlan


_PRESET_DIR = Path.home() / ".vector_kb" / "presets"


class VectorizationPlanDialog(QDialog):  # pragma: no cover - UI code
    def __init__(self, preview: PlanPreview, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.preview = preview
        self._selected_plan = preview.plan
        self.setWindowTitle("Vectorization Plan")
        self.resize(720, 600)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        tabs = QTabWidget(self)

        self.plan_editor = QPlainTextEdit(self)
        self.plan_editor.setPlainText(
            json.dumps(self.preview.plan.model_dump(), indent=2, ensure_ascii=False)
        )
        self.plan_editor.setReadOnly(False)
        tabs.addTab(self.plan_editor, "Plan (JSON)")

        preview_widget = QTextEdit(self)
        preview_widget.setReadOnly(True)
        preview_text = []
        for chunk in self.preview.chunks:
            preview_text.append(
                f"ID: {chunk.id_example}\nTokens: {chunk.tokens}\nStart: {chunk.start} End: {chunk.end}\n\n{chunk.text}\n{'-' * 40}"
            )
        preview_widget.setPlainText("\n".join(preview_text) if preview_text else "No sample chunks")
        tabs.addTab(preview_widget, "Chunk Preview")

        stats_widget = QWidget(self)
        stats_layout = QVBoxLayout(stats_widget)
        stats_table = QTableWidget(self)
        stats_table.setColumnCount(2)
        stats_table.setHorizontalHeaderLabels(["Metric", "Value"])
        stats_table.horizontalHeader().setStretchLastSection(True)
        stats_table.verticalHeader().setVisible(False)
        stats_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        stats_table.setRowCount(len(self.preview.stats))
        for row, (key, value) in enumerate(self.preview.stats.items()):
            stats_table.setItem(row, 0, QTableWidgetItem(str(key)))
            stats_table.setItem(row, 1, QTableWidgetItem(str(value)))
        stats_layout.addWidget(stats_table)
        tabs.addTab(stats_widget, "Stats & Checks")

        layout.addWidget(tabs)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        self.button_box.button(QDialogButtonBox.Ok).setText("Apply Plan")
        self.button_box.rejected.connect(self.reject)
        self.button_box.accepted.connect(self.accept)

        validate_button = self.button_box.addButton(
            "Validate & Use Edited Plan", QDialogButtonBox.ActionRole
        )
        validate_button.clicked.connect(self._on_validate_clicked)

        preset_button = self.button_box.addButton(
            "Save as Preset", QDialogButtonBox.ActionRole
        )
        preset_button.clicked.connect(self._on_save_preset)

        layout.addWidget(self.button_box)

    def plan(self) -> VectorizationPlan:
        return self._selected_plan

    # -- helpers ---------------------------------------------------------
    def _parse_plan(self) -> VectorizationPlan:
        raw = self.plan_editor.toPlainText()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Plan JSON is invalid: {exc}") from exc
        try:
            return VectorizationPlan.from_dict(payload)
        except Exception as exc:
            raise ValueError(str(exc)) from exc

    def _on_validate_clicked(self) -> None:
        try:
            plan = self._parse_plan()
        except ValueError as exc:
            QMessageBox.critical(self, "Plan Validation", str(exc))
            return
        self._selected_plan = plan
        QMessageBox.information(self, "Plan Validation", "Plan JSON is valid and ready to use.")

    def _on_save_preset(self) -> None:
        try:
            plan = self._parse_plan()
        except ValueError as exc:
            QMessageBox.critical(self, "Plan Preset", str(exc))
            return
        self._selected_plan = plan
        _PRESET_DIR.mkdir(parents=True, exist_ok=True)
        default_name = f"{plan.collection_name or 'collection'}_{plan.doc_type or 'plan'}.json"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Plan Preset",
            str(_PRESET_DIR / default_name),
            "JSON Files (*.json)",
        )
        if not path:
            return
        try:
            Path(path).write_text(
                json.dumps(plan.model_dump(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            QMessageBox.critical(self, "Plan Preset", f"Failed to write preset: {exc}")
            return
        QMessageBox.information(self, "Plan Preset", f"Saved preset to {path}")

    def accept(self) -> None:  # pragma: no cover - UI interaction
        try:
            plan = self._parse_plan()
        except ValueError as exc:
            QMessageBox.critical(self, "Plan Validation", str(exc))
            return
        self._selected_plan = plan
        super().accept()

