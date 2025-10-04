"""Logging utilities for BDlaw."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


DEFAULT_LOG_FILE = "bdlaw.log"
DEFAULT_MAX_BYTES = 5 * 1024 * 1024
DEFAULT_BACKUP_COUNT = 3


def configure_logging(
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    console: bool = True,
) -> None:
    """Configure application logging with rotation.

    Parameters
    ----------
    log_file:
        Optional path to the log file. If omitted, ``DEFAULT_LOG_FILE`` in the
        current working directory is used.
    level:
        Log level for the root logger.
    console:
        When ``True`` also attach a console handler for interactive runs.
    """

    log_path = Path(log_file or DEFAULT_LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler = RotatingFileHandler(
        log_path, maxBytes=DEFAULT_MAX_BYTES, backupCount=DEFAULT_BACKUP_COUNT
    )
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)


__all__ = ["configure_logging"]
