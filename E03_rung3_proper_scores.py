"""
E03_rung3_proper_scores.py — Ladder rung 3: Proper scores
=========================================================
Anchor double-helix ladder — rung 3 (proper scoring rules).

Uses **SurvivalEVAL** (same package as AN / E02):
IPCW-IBS (H1 primary), naive IBS, pointwise Brier at month horizons.
CRPS / censored loglik remain stub (API not on SurvivalEvaluator 0.8).

Writes ``results/ladder/d0{n}_scores.json`` per domain in E00 manifest.

Execute:
    python -W default E03_rung3_proper_scores.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import cfg
from src.metrics.calibration import make_survival_evaluator
from src.metrics.freeze import load_frozen_manifest
from src.metrics.io import utc_now, write_json
from src.metrics.predict import load_train_for_ipcw, predict_survival_curves
from src.metrics.proper_scores import (
    brier_at_horizons,
    censored_loglik,
    crps,
    ipcw_integrated_brier,
    ipcw_integrated_brier_sksurv,
    naive_integrated_brier,
)

warnings.filterwarnings("default")


def log(msg: str = "") -> None:
    print(msg, flush=True)


def _slug(domain_id: str) -> str:
    return domain_id.lower().replace("domain_", "d")


def eval_one_model(artifact: dict) -> dict:
    mid = artifact["model_id"]
    domain_id = artifact["domain_id"]
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

    backend = (artifact.get("backend") or "").lower()
    use_sksurv_ibs = "xgb" in backend or mid == "cox_xgboost"
    # Bondora×XGB Breslow @ n_grid≥100 can SIGSEGV inside SurvivalEVAL/sksurv;
    # keep a moderate grid for the stable H1 path.
    n_grid = 40 if use_sksurv_ibs else 100

    try:
        curves = predict_survival_curves(artifact, n_grid=n_grid)
        train = load_train_for_ipcw(artifact)
        train_time = train_event = None
        if train is not None:
            train_time, train_event = train

        if use_sksurv_ibs:
            # SurvivalEVAL 0.8 can SIGSEGV on large Bondora×XGB Breslow grids;
            # sksurv IPCW-IBS is the stable H1 path for this estimator.
            if train_time is None:
                return {
                    "model_id": mid,
                    "implementation_status": "error",
                    "error": "IPCW train missing for sksurv IBS",
                    "sha256": artifact.get("sha256"),
                }
            ibs_ipcw = ipcw_integrated_brier_sksurv(
                time_train=train_time,
                event_train=train_event,
                time_test=curves["time"],
                event_test=curves["event"],
                surv_grid=curves["surv_grid"],
                times_grid=curves["times_grid"],
            )
            if ibs_ipcw.get("status") != "live":
                return {
                    "model_id": mid,
                    "h1_rank_key": artifact.get("h1_rank_key"),
                    "sha256": artifact.get("sha256"),
                    "implementation_status": "error",
                    "error": ibs_ipcw.get("error") or "sksurv IBS failed",
                    "ipcw_ibs": ibs_ipcw,
                    "curve_backend": curves.get("curve_backend"),
                    "metric_backend": "sksurv",
                }
            return {
                "model_id": mid,
                "h1_rank_key": artifact.get("h1_rank_key"),
                "sha256": artifact.get("sha256"),
                "implementation_status": "live",
                "n": curves["n"],
                "n_events": curves["n_events"],
                "eval_mode": artifact.get("eval_mode"),
                "curve_backend": curves["curve_backend"],
                "metric_backend": "sksurv",
                "n_grid": n_grid,
                "ipcw_ibs": ibs_ipcw,
                "naive_ibs": {
                    "metric": "naive_ibs",
                    "status": "skipped",
                    "value": None,
                    "note": "sksurv path reports IPCW-IBS only for H1 ranking",
                },
                "brier_at_horizons": {"status": "skipped", "note": "sksurv H1 path"},
                "censored_loglik": censored_loglik(),
                "crps": crps(),
                "hypothesis_hooks": ["H1"],
                "note": "Breslow S(t) via D03b; IPCW-IBS via sksurv (SurvivalEVAL skipped; n_grid=40).",
            }

        evaluator = make_survival_evaluator(
            curves["surv_grid"],
            curves["times_grid"],
            curves["time"],
            curves["event"],
            train_time=train_time,
            train_event=train_event,
        )
        ibs_ipcw = ipcw_integrated_brier(evaluator)
        ibs_naive = naive_integrated_brier(evaluator)
        brier_h = brier_at_horizons(evaluator, domain_id=domain_id)
        return {
            "model_id": mid,
            "h1_rank_key": artifact.get("h1_rank_key"),
            "sha256": artifact.get("sha256"),
            "implementation_status": "live",
            "n": curves["n"],
            "n_events": curves["n_events"],
            "eval_mode": artifact.get("eval_mode"),
            "curve_backend": curves["curve_backend"],
            "metric_backend": "SurvivalEVAL",
            "ipcw_ibs": ibs_ipcw,
            "naive_ibs": ibs_naive,
            "brier_at_horizons": brier_h,
            "censored_loglik": censored_loglik(),
            "crps": crps(),
            "hypothesis_hooks": ["H1"],
        }
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


def _is_xgb_artifact(artifact: dict) -> bool:
    backend = (artifact.get("backend") or "").lower()
    mid = artifact.get("model_id") or ""
    return "xgb" in backend or mid == "cox_xgboost"


def _eval_xgb_in_subprocess(artifact: dict) -> dict:
    """
    SurvivalEVAL C-extensions leave process state that makes a later
    ``joblib.load`` of an XGB Booster SIGSEGV — even after mp.spawn in some
    envs. Run XGB IBS in a brand-new ``python`` interpreter.
    """
    import json
    import subprocess
    import tempfile

    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="e03_xgb_") as tmp:
        art_path = Path(tmp) / "artifact.json"
        out_path = Path(tmp) / "result.json"
        art_path.write_text(json.dumps(artifact), encoding="utf-8")
        worker = f"""
import json, sys
from pathlib import Path
sys.path.insert(0, {str(root)!r})
from E03_rung3_proper_scores import eval_one_model
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
                "error": f"XGB IBS subprocess rc={proc.returncode}: {err}",
                "sha256": artifact.get("sha256"),
                "note": "Fresh python -c isolation after SurvivalEVAL.",
            }
        result = json.loads(out_path.read_text(encoding="utf-8"))
        if isinstance(result, dict) and result.get("implementation_status") == "live":
            result = {
                **result,
                "note": (
                    (result.get("note") or "")
                    + " [eval via fresh python subprocess]"
                ).strip(),
            }
        return result


def eval_one_model_safe(artifact: dict) -> dict:
    """Prefer in-process eval; isolate XGB after SurvivalEVAL corruption risk."""
    if _is_xgb_artifact(artifact):
        return _eval_xgb_in_subprocess(artifact)
    return eval_one_model(artifact)


def main() -> int:
    log("─" * 60)
    log("E03 — RUNG 3 PROPER SCORES (SurvivalEVAL IPCW-IBS)")
    log("─" * 60)

    manifest = load_frozen_manifest()
    by_domain: dict[str, list] = {}
    for a in manifest["artifacts"]:
        if not a.get("present"):
            continue
        by_domain.setdefault(a["domain_id"], []).append(a)

    out_paths = []
    for domain_id, arts in sorted(by_domain.items()):
        models = [eval_one_model_safe(a) for a in arts]
        n_live = sum(1 for m in models if m.get("implementation_status") == "live")
        payload = {
            "stage": "E03",
            "rung": 3,
            "rung_name": "proper_scores",
            "anchor_parallel": "double-helix ladder rung 3 — proper scores",
            "generated_at_utc": utc_now(),
            "domain_id": domain_id,
            "implementation_status": "live" if n_live else "partial",
            "n_live_models": n_live,
            "models": models,
            "hypothesis_hooks": ["H1"],
            "metric_backend": "SurvivalEVAL+sksurv",
            "rule": (
                "No retraining — frozen D artifacts only; absolute S(t|x) required; "
                "IBS via SurvivalEVAL (Cox/RSF) or sksurv (XGB Breslow, n_grid=40, subprocess)."
            ),
        }
        path = cfg.DIRS["ladder"] / f"{_slug(domain_id)}_scores.json"
        write_json(path, payload)
        out_paths.append(path)
        log(f"  {domain_id}: live_models={n_live}")
        for m in models:
            st = m.get("implementation_status")
            if st == "live":
                ipcw = (m.get("ipcw_ibs") or {}).get("value")
                naive = (m.get("naive_ibs") or {}).get("value")
                bh = (m.get("brier_at_horizons") or {}).get("values") or {}
                ipcw_s = f"{ipcw:.4f}" if ipcw is not None else "NA"
                naive_s = f"{naive:.4f}" if naive is not None else "NA"
                h_s = ", ".join(
                    f"{k}m={v:.4f}" if v is not None else f"{k}m=NA"
                    for k, v in bh.items()
                )
                log(f"    {m['model_id']}: IPCW-IBS={ipcw_s}  naive-IBS={naive_s}  Brier[{h_s}]")
            else:
                log(f"    {m['model_id']}: {st}")

    log(f"E03 complete — {len(out_paths)} domain file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
