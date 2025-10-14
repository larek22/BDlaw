"""Qt dialog that previews vectorization plans."""
from __future__ import annotations

import json
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QPlainTextEdit,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..vectorization.models import PlanPreview, VectorizationPlan


class VectorizationPlanDialog(QDialog):  # pragma: no cover - UI code
    def __init__(self, preview: PlanPreview, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.preview = preview
        self.setWindowTitle("Vectorization Plan")
        self.resize(720, 600)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        tabs = QTabWidget(self)

        plan_editor = QPlainTextEdit(self)
        plan_editor.setPlainText(json.dumps(self.preview.plan.model_dump(), indent=2, ensure_ascii=False))
        plan_editor.setReadOnly(True)
        tabs.addTab(plan_editor, "Plan (JSON)")

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
        layout.addWidget(self.button_box)

    def plan(self) -> VectorizationPlan:
        return self.preview.plan

