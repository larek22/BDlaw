from __future__ import annotations

from app.logging_config import configure_logging
from app.settings import AppSettings
from app.ui.main_window import run_app


def main() -> None:  # pragma: no cover
    configure_logging()
    run_app()


if __name__ == "__main__":  # pragma: no cover
    main()
