"""
E00_load_frozen_models.py — Gate: inventory + hash + sanity-load Block D
=======================================================================
Not a ladder rung — prepares frozen artifacts for E01–E05 (anchor rungs).

Writes ``results/ladder/frozen_models_manifest.json`` with, per artifact:
  - sha256 / present / predict_ready
  - eval_data_path, time_col, event_col, eval_mode
  - load_ok (joblib sanity load for estimators; no predict / no retrain)

Rules:
  - Requires C00 ``protocol_freeze.json`` (hard fail).
  - Soft-warns if D99 ``report.json`` is missing.
  - Optional models (H4 ablated) may be absent.
  - DOMAIN_03 stays cv_scores-only until an estimator is exported.

Re-run after any Domain lane refresh so rung scripts see updated hashes.

Execute:
    python -W default E00_load_frozen_models.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import cfg
from src.metrics.freeze import (
    FROZEN_MANIFEST_NAME,
    inventory_artifacts,
    require_protocol_freeze,
)
from src.metrics.io import utc_now, write_json

warnings.filterwarnings("default")

# Anchor double-helix ladder (PIPELINE / Lillelund) — for manifest metadata
LADDER_RUNGS = [
    {"id": "E01", "rung": 1, "script": "E01_rung1_discrimination.py", "name": "discrimination"},
    {"id": "E02", "rung": 2, "script": "E02_rung2_calibration.py", "name": "calibration"},
    {"id": "E03", "rung": 3, "script": "E03_rung3_proper_scores.py", "name": "proper_scores"},
    {"id": "E04", "rung": 4, "script": "E04_rung4_dependent_censoring.py", "name": "dependent_censoring"},
    {"id": "E05", "rung": 5, "script": "E05_rung5_competing_risks.py", "name": "competing_risks"},
]


def log(msg: str = "") -> None:
    print(msg, flush=True)


def main() -> int:
    log("─" * 60)
    log("E00 — LOAD / HASH FROZEN MODELS (ladder gate)")
    log("─" * 60)

    protocol = require_protocol_freeze()
    log(
        f"  Protocol freeze OK  version={protocol.get('protocol_version')}  "
        f"sha={str(protocol.get('content_sha256', ''))[:12]}…"
    )

    d99 = cfg.DIRS["reproduction"] / "report.json"
    if d99.exists():
        log(f"  D99 report present: {d99.relative_to(cfg.ROOT)}")
        d99_status = "present"
    else:
        log(
            f"  WARN: D99 report missing ({d99.relative_to(cfg.ROOT)}) — "
            "OK to develop E; close Fase A before claiming Results."
        )
        d99_status = "missing"

    log("  Sanity-loading estimators (joblib)…")
    artifacts = inventory_artifacts(cfg.ROOT, sanity_load=True)

    missing_required = [
        a for a in artifacts if a["required_for_e00"] and not a["present"]
    ]
    load_failed = [
        a
        for a in artifacts
        if a["required_for_e00"]
        and a["present"]
        and a["kind"] == "estimator"
        and a.get("load_ok") is False
    ]

    for a in artifacts:
        if not a["present"]:
            flag = "MISS*" if a["required_for_e00"] else "opt"
        elif a["kind"] == "estimator" and a.get("load_ok") is False:
            flag = "LOAD!"
        elif a.get("predict_ready"):
            flag = "OK"
        elif a["kind"] == "cv_scores":
            flag = "CV"
        else:
            flag = "ok"
        sha = (a["sha256"] or "—")[:12]
        nfeat = a.get("n_features")
        feat_s = f"  p={nfeat}" if nfeat is not None else ""
        eval_ok = "eval+" if a.get("eval_data_present") else "eval-"
        log(
            f"  [{flag}] {a['domain_id']}/{a['model_id']}  "
            f"sha={sha}…  {eval_ok}{feat_s}"
        )
        if a.get("load_error"):
            log(f"         load_error: {a['load_error']}")

    if missing_required:
        names = [f"{a['domain_id']}/{a['model_id']}" for a in missing_required]
        log(f"E00 blocked — missing required artifacts: {names}")
        return 1
    if load_failed:
        names = [f"{a['domain_id']}/{a['model_id']}" for a in load_failed]
        log(f"E00 blocked — joblib load failed: {names}")
        return 1

    # Do not serialize full feature lists into a huge manifest if very long —
    # keep them (E01 needs them); Bondora ~70 feats is fine.
    cfg.DIRS["ladder"].mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": "E00",
        "artifact": "frozen_models_manifest",
        "generated_at_utc": utc_now(),
        "protocol_version": protocol.get("protocol_version"),
        "protocol_content_sha256": protocol.get("content_sha256"),
        "d99_report": d99_status,
        "ladder_rungs": LADDER_RUNGS,
        "n_artifacts": len(artifacts),
        "n_present": sum(1 for a in artifacts if a["present"]),
        "n_predict_ready": sum(1 for a in artifacts if a.get("predict_ready")),
        "n_load_ok": sum(1 for a in artifacts if a.get("load_ok")),
        "h1_rankings": protocol.get("freeze_decisions", {})
        .get("C00.2", {})
        .get("rankings", {}),
        "artifacts": artifacts,
        "rule": (
            "Block E must not retrain baselines. "
            "Re-run E00 after D refreshes; then re-run E01–E05 rungs."
        ),
    }
    out = cfg.DIRS["ladder"] / FROZEN_MANIFEST_NAME
    write_json(out, manifest)
    log(f"  Wrote {out.relative_to(cfg.ROOT)}")
    log(
        f"E00 complete — {manifest['n_present']}/{manifest['n_artifacts']} present, "
        f"{manifest['n_predict_ready']} predict-ready, "
        f"{manifest['n_load_ok']} load_ok."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
