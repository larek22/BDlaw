"""Entrypoint for launching the BDlaw desktop UI via ``python -m``."""

from __future__ import annotations

from bdlaw.app.ui.main import run_ui
from bdlaw.settings.config import load_config


def main() -> None:
    config = load_config()
    run_ui(config)


if __name__ == "__main__":  # pragma: no cover - executable module
    main()
