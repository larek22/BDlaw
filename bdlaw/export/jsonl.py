"""Export utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from bdlaw.parser.structure import Norm


def export_norms_to_jsonl(norms: Iterable[Norm], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for norm in norms:
            fh.write(json.dumps(norm.__dict__, ensure_ascii=False) + "\n")


__all__ = ["export_norms_to_jsonl"]
