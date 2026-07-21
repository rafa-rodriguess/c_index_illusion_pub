"""JSON / Markdown helpers for ``results/ladder/`` artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stub_payload(
    *,
    stage: str,
    rung: str,
    message: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "stage": stage,
        "rung": rung,
        "implementation_status": "stub",
        "generated_at_utc": utc_now(),
        "message": message,
        "rule": "No retraining in Block E — predict / re-score frozen D artifacts only.",
    }
    if extra:
        doc.update(extra)
    return doc
