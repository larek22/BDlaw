from pathlib import Path
from typing import List

from PySide6 import QtWidgets

from bdlaw.app.pipeline import ApplicationPipelines
from bdlaw.app.ui.main import run_ui
from bdlaw.parser.structure import ParsingContext
from bdlaw.settings.config import load_config


def main() -> None:
    config = load_config()
    pipelines = ApplicationPipelines(config)

    def handle_import(paths: List[Path]) -> None:
        if not paths:
            return
        parent = QtWidgets.QApplication.activeWindow()
        law_code, ok = QtWidgets.QInputDialog.getText(parent, "Код закона", "Например: ГК РФ")
        if not ok or not law_code:
            return
        version_id, ok = QtWidgets.QInputDialog.getText(parent, "Версия", "ГГГГ-ММ-ДД")
        if not ok or not version_id:
            return
        valid_from = version_id
        context = ParsingContext(
            jurisdiction="ru",
            act_type="federal_law",
            law_code=law_code,
            law_full_title=law_code,
            version_id=version_id,
            valid_from=valid_from,
        )
        parsed = pipelines.ingest_paths(paths, context)
        pipelines.index_parsed(parsed)
        QtWidgets.QMessageBox.information(parent, "Импорт завершён", f"Индексировано норм: {len(parsed.norms)}")

    run_ui(config, handle_import)


if __name__ == "__main__":
    main()
