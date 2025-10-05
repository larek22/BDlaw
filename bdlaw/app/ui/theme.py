"""Application-wide styling helpers for the BDlaw desktop UI."""

from __future__ import annotations

from PySide6 import QtGui, QtWidgets


PRIMARY_BG = QtGui.QColor(22, 24, 28)
SECONDARY_BG = QtGui.QColor(36, 39, 45)
ACCENT_COLOR = QtGui.QColor(80, 140, 255)
TEXT_COLOR = QtGui.QColor(235, 237, 240)
SUBTEXT_COLOR = QtGui.QColor(180, 184, 189)


def _build_palette() -> QtGui.QPalette:
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, PRIMARY_BG)
    palette.setColor(QtGui.QPalette.WindowText, TEXT_COLOR)
    palette.setColor(QtGui.QPalette.Base, SECONDARY_BG)
    palette.setColor(QtGui.QPalette.AlternateBase, PRIMARY_BG.darker(115))
    palette.setColor(QtGui.QPalette.ToolTipBase, SECONDARY_BG)
    palette.setColor(QtGui.QPalette.ToolTipText, TEXT_COLOR)
    palette.setColor(QtGui.QPalette.Text, TEXT_COLOR)
    palette.setColor(QtGui.QPalette.Button, SECONDARY_BG)
    palette.setColor(QtGui.QPalette.ButtonText, TEXT_COLOR)
    palette.setColor(QtGui.QPalette.Highlight, ACCENT_COLOR)
    palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("white"))
    palette.setColor(QtGui.QPalette.PlaceholderText, SUBTEXT_COLOR)
    return palette


STYLE_SHEET = """
QMainWindow, QWidget {
    background-color: #16181c;
    color: #ebedf0;
    font-family: "Segoe UI", "SF Pro Display", "Roboto", sans-serif;
    font-size: 14px;
}

QLabel#PageTitle {
    font-size: 22px;
    font-weight: 600;
    margin-bottom: 12px;
}

QPushButton {
    background-color: #24272d;
    border: 1px solid #2f333a;
    border-radius: 6px;
    padding: 10px 14px;
    color: #ebedf0;
}

QPushButton:hover {
    background-color: #2d3138;
}

QPushButton:pressed {
    background-color: #202327;
}

QPushButton:disabled {
    background-color: #1d1f23;
    color: #70757d;
    border-color: #1d1f23;
}

QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QComboBox {
    background-color: #1f2227;
    border: 1px solid #32363d;
    border-radius: 6px;
    padding: 6px 8px;
    selection-background-color: #508cff;
    selection-color: white;
}

QListWidget, QTreeWidget, QTableWidget {
    background-color: #1f2227;
    border: 1px solid #30343b;
    border-radius: 6px;
}

QProgressBar {
    border: 1px solid #30343b;
    border-radius: 6px;
    background: #1d1f23;
    text-align: center;
}

QProgressBar::chunk {
    background-color: #508cff;
    border-radius: 6px;
}

QToolBar {
    background: #16181c;
    border-bottom: 1px solid #22252b;
    spacing: 8px;
}

QToolButton {
    padding: 6px 10px;
}

QSplitter::handle {
    background: #2b2f36;
    width: 2px;
}

QScrollBar:vertical {
    background: #1d1f23;
    width: 12px;
    margin: 4px;
}

QScrollBar::handle:vertical {
    background: #2d3138;
    border-radius: 6px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background: #3a3f47;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
"""


def apply_dark_theme(app: QtWidgets.QApplication) -> None:
    """Install the BDlaw dark palette and stylesheet on *app*."""

    app.setStyle("Fusion")
    app.setPalette(_build_palette())
    app.setStyleSheet(STYLE_SHEET)


__all__ = ["apply_dark_theme"]
