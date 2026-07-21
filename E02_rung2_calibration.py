"""
E02_rung2_calibration.py — Ladder rung 2: Calibration
=====================================================
Anchor double-helix ladder — rung 2 (calibration).

Uses **SurvivalEVAL** (same package as AN harness / Lillelund et al.):
D-Calibration, Austin ICI, KM-calibration on frozen D estimators with
absolute S(t|x) (lifelines Cox; XGB Cox via D03b Breslow, n_grid=40).
D3 CV scores → skipped_no_estimator.

Writes ``results/ladder/d0{n}_calibration.json`` per domain in E00 manifest.

Execute:
    python -W default E02_rung2_calibration.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import cfg
from src.metrics.calibration import (
    d_calibration,
    integrated_calibration_index,
    km_vs_predicted,
    make_survival_evaluator,
)
from src.metrics.freeze import load_frozen_manifest
from src.metrics.io import utc_now, write_json
from src.metrics.predict import load_train_for_ipcw, predict_survival_curves

warnings.filterwarnings("default")


def log(msg: str = "") -> None:
    print(msg, flush=True)


def _slug(domain_id: str) -> str:
    return domain_id.lower().replace("domain_", "d")


def _is_xgb_artifact(artifact: dict) -> bool:
    backend = (artifact.get("backend") or "").lower()
    mid = artifact.get("model_id") or ""
    return "xgb" in backend or mid == "cox_xgboost"


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

    use_xgb = _is_xgb_artifact(artifact)
    # Same grid cap as E03 — SurvivalEVAL can SIGSEGV on Bondora×XGB @ n_grid=100.
    n_grid = 40 if use_xgb else 100

    try:
        curves = predict_survival_curves(artifact, n_grid=n_grid)
        train = load_train_for_ipcw(artifact)
        train_time = train_event = None
        if train is not None:
            train_time, train_event = train

        evaluator = make_survival_evaluator(
            curves["surv_grid"],
            curves["times_grid"],
            curves["time"],
            curves["event"],
            train_time=train_time,
            train_event=train_event,
        )
        dcal = d_calibration(evaluator)
        ici = integrated_calibration_index(evaluator)
        km = km_vs_predicted(evaluator)
        out = {
            "model_id": mid,
            "h1_rank_key": artifact.get("h1_rank_key"),
            "sha256": artifact.get("sha256"),
            "implementation_status": "live",
            "n": curves["n"],
            "n_events": curves["n_events"],
            "eval_mode": artifact.get("eval_mode"),
            "curve_backend": curves["curve_backend"],
            "metric_backend": "SurvivalEVAL",
            "n_grid": n_grid,
            "d_calibration": dcal,
            "ici": ici,
            "km_vs_predicted": km,
            "hypothesis_hooks": ["H2"],
        }
        if use_xgb:
            out["note"] = "Breslow S(t) via D03b; SurvivalEVAL @ n_grid=40."
        return out
    except TypeError as exc:
        return {
            "model_id": mid,
            "implementation_status": "skipped_no_survival_curve",
            "note": str(exc),
            "sha256": artifact.get("sha256"),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "model_id": mid,
            "implementation_status": "error",
            "error": str(exc),
            "sha256": artifact.get("sha256"),
        }


def _eval_xgb_in_subprocess(artifact: dict) -> dict:
    """Fresh interpreter: SurvivalEVAL before joblib.load(XGB) SIGSEGVs here."""
    import json
    import subprocess
    import tempfile

    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="e02_xgb_") as tmp:
        art_path = Path(tmp) / "artifact.json"
        out_path = Path(tmp) / "result.json"
        art_path.write_text(json.dumps(artifact), encoding="utf-8")
        worker = f"""
import json, sys
from pathlib import Path
sys.path.insert(0, {str(root)!r})
from E02_rung2_calibration import eval_one_model
art = json.loads(Path({str(art_path)!r}).read_text(encoding="utf-8"))
res = eval_one_model(art)
Path({str(out_path)!r}).write_text(json.dumps(res), encoding="utf-8")
"""
        proc = subprocess.run(
            [sys.executable, "-c", worker],
            cwd=str(root),
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "")[-800:]
            return {
                "model_id": artifact.get("model_id"),
                "implementation_status": "error",
                "error": f"XGB D-Cal subprocess rc={proc.returncode}: {err}",
                "sha256": artifact.get("sha256"),
                "note": "Fresh python -c isolation after SurvivalEVAL.",
            }
        result = json.loads(out_path.read_text(encoding="utf-8"))
        if isinstance(result, dict) and result.get("implementation_status") == "live":
            note = (result.get("note") or "").strip()
            result["note"] = (note + " [eval via fresh python subprocess]").strip()
        return result


def eval_one_model_safe(artifact: dict) -> dict:
    if _is_xgb_artifact(artifact):
        return _eval_xgb_in_subprocess(artifact)
    return eval_one_model(artifact)


def main() -> int:
    log("─" * 60)
    log("E02 — RUNG 2 CALIBRATION (SurvivalEVAL D-Cal / ICI / KM)")
    log("─" * 60)

    manifest = load_frozen_manifest()
    by_domain: dict[str, list] = {}
    for a in manifest["artifacts"]:
        if not a.get("present"):
            continue
        by_domain.setdefault(a["domain_id"], []).append(a)

    out_paths = []
    for domain_id, arts in sorted(by_domain.items()):
        # XGB first within domain so classical SurvivalEVAL never precedes Booster load
        # in-process; XGB itself still runs in a fresh subprocess.
        arts_sorted = sorted(
            arts, key=lambda a: 0 if _is_xgb_artifact(a) else 1
        )
        models = [eval_one_model_safe(a) for a in arts_sorted]
        n_live = sum(1 for m in models if m.get("implementation_status") == "live")
        payload = {
            "stage": "E02",
            "rung": 2,
            "rung_name": "calibration",
            "anchor_parallel": "double-helix ladder rung 2 — calibration",
            "generated_at_utc": utc_now(),
            "domain_id": domain_id,
            "implementation_status": "live" if n_live else "partial",
            "n_live_models": n_live,
            "models": models,
            "hypothesis_hooks": ["H2"],
            "metric_backend": "SurvivalEVAL",
            "rule": (
                "No retraining — frozen D artifacts only; absolute S(t|x) required; "
                "metrics via SurvivalEVAL (anchor stack)."
            ),
        }
        path = cfg.DIRS["ladder"] / f"{_slug(domain_id)}_calibration.json"
        write_json(path, payload)
        out_paths.append(path)
        log(f"  {domain_id}: live_models={n_live}")
        for m in models:
            st = m.get("implementation_status")
            if st == "live":
                p = (m.get("d_calibration") or {}).get("p_value")
                rej = (m.get("d_calibration") or {}).get("reject_h0_well_calibrated")
                ici = (m.get("ici") or {}).get("value")
                kmv = (m.get("km_vs_predicted") or {}).get("value")
                p_s = f"{p:.4g}" if p is not None else "NA"
                ici_s = f"{ici:.4f}" if ici is not None else "NA"
                km_s = f"{kmv:.4f}" if kmv is not None else "NA"
                log(
                    f"    {m['model_id']}: D-Cal p={p_s} reject={rej}  "
                    f"ICI={ici_s}  KM-cal={km_s}"
                )
            else:
                log(f"    {m['model_id']}: {st}")

    log(f"E02 complete — {len(out_paths)} domain file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
