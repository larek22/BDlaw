"""Convenient launcher for the BDlaw desktop application."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import traceback
from pathlib import Path


def _ensure_qdrant(compose: bool) -> None:
    if not compose:
        print("Автозапуск Qdrant отключён (--no-compose). Предполагается, что сервис уже запущен.")
        return

    compose_file = Path(__file__).with_name("docker-compose.yml")
    if not compose_file.exists():
        print("docker-compose.yml не найден. Пропускаем автозапуск Qdrant.")
        return

    docker_bin = shutil.which("docker")
    if docker_bin is None:
        print("Docker не найден в PATH. Пропускаем автозапуск Qdrant. Запустите Qdrant вручную.")
        return

    try:
        subprocess.run(
            [docker_bin, "compose", "up", "-d", "qdrant"],
            check=True,
            cwd=compose_file.parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        print("Qdrant запущен через docker compose.")
    except subprocess.CalledProcessError as exc:  # pragma: no cover - external command
        stderr = exc.stderr.decode(errors="ignore")
        print(
            "Не удалось автоматически запустить Qdrant через docker compose. "
            "Запустите сервис вручную и повторите попытку.\n"
            f"Сообщение docker: {stderr.strip()}"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Быстрый запуск BDlaw GUI")
    parser.add_argument(
        "--no-compose",
        action="store_true",
        help="не пытаться автоматически поднять docker-compose сервис Qdrant",
    )
    args = parser.parse_args(argv)

    _ensure_qdrant(not args.no_compose)

    try:
        from bdlaw.settings.config import load_config
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing == "bdlaw":
            raise
        _print_error(
            "Не удалось импортировать зависимости приложения. "
            "Убедитесь, что виртуальное окружение активировано и выполнена команда"
            " 'pip install -e .[ui]'.\n"
            f"Отсутствующий пакет: {missing or exc}"
        )
        sys.exit(1)

    try:
        from bdlaw.app.ui import run_ui
    except ModuleNotFoundError as exc:
        missing = exc.name or "PySide6"
        _print_error(
            "Не удалось запустить графический интерфейс: отсутствует зависимость"
            f" '{missing}'.\n"
            "Установите её командой 'pip install PySide6' (или активируйте окружение,"
            " где библиотека уже установлена) и повторите запуск."
        )
        sys.exit(1)

    try:
        config = load_config()
        run_ui(config)
    except Exception as exc:  # noqa: BLE001 - surface message before closing console
        _print_error(
            "Не удалось запустить графический интерфейс. Подробности приведены ниже:\n"
            f"{exc}\n\n{traceback.format_exc()}"
        )
        sys.exit(1)


def _print_error(message: str) -> None:
    print(message)
    if sys.stdin.isatty():
        try:
            input("Нажмите Enter, чтобы закрыть окно…")
        except EOFError:
            pass


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    main(sys.argv[1:])
