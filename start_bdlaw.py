"""Convenient launcher for the BDlaw desktop application."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _ensure_qdrant(compose: bool) -> None:
    if not compose:
        return

    compose_file = Path(__file__).with_name("docker-compose.yml")
    if not compose_file.exists():
        raise SystemExit("docker-compose.yml не найден — убедитесь, что запускаете из корня проекта")

    try:
        subprocess.run(
            ["docker", "compose", "up", "-d", "qdrant"],
            check=True,
            cwd=compose_file.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:  # pragma: no cover - depends on docker availability
        raise SystemExit("Не удалось найти Docker. Установите Docker Desktop или запустите Qdrant вручную") from exc
    except subprocess.CalledProcessError as exc:  # pragma: no cover - external command
        raise SystemExit(f"Запуск Qdrant через docker compose завершился с ошибкой:\n{exc.stderr.decode()}" ) from exc


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Быстрый запуск BDlaw GUI")
    parser.add_argument(
        "--no-compose",
        action="store_true",
        help="не пытаться автоматически поднять docker-compose сервис Qdrant",
    )
    args = parser.parse_args(argv)

    _ensure_qdrant(not args.no_compose)

    from bdlaw.settings.config import load_config
    from bdlaw.app.ui import run_ui

    config = load_config()
    run_ui(config)


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main(sys.argv[1:])
