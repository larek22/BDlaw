#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from app.data_repository import DataRepository
from app.logging_config import configure_logging
from app.qdrant_client import QdrantVectorStore
from app.settings import AppSettings
from verify import run_verification


def _create_snapshot(vector_store: QdrantVectorStore, output: Path | None) -> Path:
    client = vector_store.client
    response = client.create_snapshot(
        collection_name=vector_store.collection_name,
        wait=True,
    )
    snapshot_name = getattr(response, "name", None)
    if not snapshot_name:
        result = getattr(response, "result", None)
        if isinstance(result, dict):
            snapshot_name = result.get("name")
    if not snapshot_name:
        raise RuntimeError("Qdrant did not return a snapshot name")

    destination_dir = output or DataRepository().paths.snapshots
    destination_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    filename = f"{vector_store.collection_name}-{timestamp}.snapshot"
    destination = destination_dir / filename
    client.download_snapshot(
        collection_name=vector_store.collection_name,
        snapshot_name=snapshot_name,
        save_path=str(destination),
    )
    return destination


def _restore_snapshot(vector_store: QdrantVectorStore, snapshot_path: Path) -> None:
    if not snapshot_path.exists():
        raise FileNotFoundError(snapshot_path)
    client = vector_store.client
    client.upload_snapshot(
        collection_name=vector_store.collection_name,
        snapshot_path=str(snapshot_path),
        wait=True,
    )
    result = run_verification(vector_store.settings)
    if not result.success:
        raise RuntimeError(
            "Snapshot restore completed but verification failed: "
            + "; ".join(result.failures)
        )
    if result.warnings:
        for warning in result.warnings:
            print(f"WARNING: {warning}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage Qdrant collection snapshots.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    create_parser = subparsers.add_parser("create", help="Create and download a snapshot")
    create_parser.add_argument(
        "--output",
        type=Path,
        help="Optional destination directory for the snapshot file",
    )

    restore_parser = subparsers.add_parser("restore", help="Upload and restore a snapshot")
    restore_parser.add_argument("snapshot", type=Path, help="Path to the snapshot file to restore")

    args = parser.parse_args()

    configure_logging()
    settings = AppSettings.load()
    vector_store = QdrantVectorStore(settings)

    try:
        if args.command == "create":
            destination = _create_snapshot(vector_store, args.output)
            print(destination)
        elif args.command == "restore":
            _restore_snapshot(vector_store, args.snapshot)
            print(f"Restored snapshot from {args.snapshot}")
        else:  # pragma: no cover - argparse enforces valid commands
            parser.error("Unknown command")
    except Exception as exc:  # pragma: no cover - runtime failures
        print(f"Snapshot operation failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
