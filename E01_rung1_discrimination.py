"""
E01_rung1_discrimination.py — Ladder rung 1: Discrimination
===========================================================
Anchor double-helix ladder — rung 1 (discrimination).

Computes Harrell C and Uno IPCW C on frozen D estimators (no retrain).
Antolini remains stub until survival-curve predictions are wired.

Writes ``results/ladder/d0{n}_discrimination.json`` per domain in E00 manifest.

Execute:
    python -W default E01_rung1_discrimination.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import cfg
from src.metrics.discrimination import (
    cindex_antolini,
    cindex_harrell,
    cindex_uno,
    multiverse_strip,
)
from src.metrics.freeze import load_frozen_manifest
from src.metrics.io import utc_now, write_json
from src.metrics.predict import load_train_for_ipcw, predict_risk_scores

warnings.filterwarnings("default")


def log(msg: str = "") -> None:
    print(msg, flush=True)


def _slug(domain_id: str) -> str:
    return domain_id.lower().replace("domain_", "d")


def eval_one_model(artifact: dict) -> dict:
    mid = artifact["model_id"]
    if artifact.get("kind") == "cv_scores":
        return {
            "model_id": mid,
            "implementation_status": "skipped_no_estimator",
            "note": artifact.get("notes"),
            "sha256": artifact.get("sha256"),
        }
    if not artifact.get("predict_ready") or not artifact.get("load_ok"):
        return {
            "model_id": mid,
            "implementation_status": "skipped_not_ready",
            "sha256": artifact.get("sha256"),
        }

    try:
        pred = predict_risk_scores(artifact)
        train = load_train_for_ipcw(artifact)
        if train is None:
            train_time = train_event = None
        else:
            train_time, train_event = train

        h = cindex_harrell(pred["time"], pred["event"], pred["risk"])
        u = cindex_uno(
            pred["time"],
            pred["event"],
            pred["risk"],
            train_time=train_time,
            train_event=train_event,
        )
        a = cindex_antolini()
        mv = multiverse_strip(
            pred["time"],
            pred["event"],
            pred["risk"],
            train_time=train_time,
            train_event=train_event,
        )
        return {
            "model_id": mid,
            "h1_rank_key": artifact.get("h1_rank_key"),
            "sha256": artifact.get("sha256"),
            "implementation_status": "live",
            "n": pred["n"],
            "n_events": pred["n_events"],
            "eval_mode": artifact.get("eval_mode"),
            "harrell": h,
            "uno": u,
            "antolini": a,
            "multiverse": mv,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "model_id": mid,
            "implementation_status": "error",
            "error": str(exc),
            "sha256": artifact.get("sha256"),
        }


def main() -> int:
    log("─" * 60)
    log("E01 — RUNG 1 DISCRIMINATION (Harrell / Uno)")
    log("─" * 60)

    manifest = load_frozen_manifest()
    by_domain: dict[str, list] = {}
    for a in manifest["artifacts"]:
        if not a.get("present"):
            continue
        by_domain.setdefault(a["domain_id"], []).append(a)

    out_paths = []
    for domain_id, arts in sorted(by_domain.items()):
        models = [eval_one_model(a) for a in arts]
        n_live = sum(1 for m in models if m.get("implementation_status") == "live")
        payload = {
            "stage": "E01",
            "rung": 1,
            "rung_name": "discrimination",
            "anchor_parallel": "double-helix ladder rung 1 — discrimination",
            "generated_at_utc": utc_now(),
            "domain_id": domain_id,
            "implementation_status": "live" if n_live else "partial",
            "n_live_models": n_live,
            "models": models,
            "rule": "No retraining — frozen D artifacts only.",
        }
        path = cfg.DIRS["ladder"] / f"{_slug(domain_id)}_discrimination.json"
        write_json(path, payload)
        out_paths.append(path)
        log(f"  {domain_id}: live_models={n_live}")
        for m in models:
            st = m.get("implementation_status")
            if st == "live":
                hv = (m.get("harrell") or {}).get("value")
                uv = (m.get("uno") or {}).get("value")
                log(f"    {m['model_id']}: Harrell={hv:.4f}  Uno={uv if uv is None else f'{uv:.4f}'}")
            else:
                log(f"    {m['model_id']}: {st}")

    log(f"E01 complete — {len(out_paths)} domain file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
