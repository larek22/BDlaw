"""Reusable dialogs for the BDlaw desktop application."""

from __future__ import annotations

from typing import Dict, Optional

from PySide6 import QtWidgets

from bdlaw.parser.structure import ParsingContext


class MetadataDialog(QtWidgets.QDialog):
    """Collect metadata required for parsing and indexing."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, defaults: Optional[Dict[str, str]] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Метаданные закона")
        self._defaults = defaults or {}
        self._fields: Dict[str, QtWidgets.QLineEdit] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        form = QtWidgets.QFormLayout()
        self.setLayout(form)

        def add_field(key: str, label: str, required: bool = False) -> None:
            edit = QtWidgets.QLineEdit(self)
            if value := self._defaults.get(key):
                edit.setText(value)
            if required:
                edit.setPlaceholderText("Обязательно")
            form.addRow(label, edit)
            self._fields[key] = edit

        add_field("jurisdiction", "Юрисдикция", required=True)
        add_field("act_type", "Тип акта", required=True)
        add_field("law_code", "Код закона", required=True)
        add_field("law_full_title", "Полное название", required=True)
        add_field("version_id", "Идентификатор редакции", required=True)
        add_field("valid_from", "Действует с", required=True)
        add_field("valid_to", "Действует до")
        add_field("supersedes", "Замещает редакцию")
        add_field("source_url", "Ссылка на источник")

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _value(self, key: str) -> str:
        return self._fields[key].text().strip()

    def get_context(self) -> ParsingContext:
        """Return :class:`ParsingContext` based on current inputs."""

        return ParsingContext(
            jurisdiction=self._value("jurisdiction") or "ru",
            act_type=self._value("act_type") or "federal_law",
            law_code=self._value("law_code"),
            law_full_title=self._value("law_full_title"),
            version_id=self._value("version_id"),
            valid_from=self._value("valid_from"),
            valid_to=self._value("valid_to") or None,
            supersedes=self._value("supersedes") or None,
            source_url=self._value("source_url") or None,
        )

    def accept(self) -> None:  # pragma: no cover - UI validation
        required = [
            ("jurisdiction", "Укажите юрисдикцию"),
            ("act_type", "Укажите тип акта"),
            ("law_code", "Введите код закона"),
            ("law_full_title", "Введите название закона"),
            ("version_id", "Укажите идентификатор редакции"),
            ("valid_from", "Укажите дату начала действия"),
        ]
        for key, message in required:
            if not self._value(key):
                QtWidgets.QMessageBox.warning(self, "Недостаточно данных", message)
                return
        super().accept()


__all__ = ["MetadataDialog"]
