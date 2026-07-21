"""
E04_rung4_dependent_censoring.py — Ladder rung 4: Dependent censoring
=====================================================================
Anchor double-helix ladder — rung 4 (dependent censoring sensitivity).

Uses **SurvivalEVAL CopulaGraphic** (Clayton) on a Kendall-τ grid matching
the anchor scenarios (0 / 0.25 / 0.50 / 0.75). Declares non-identifiability.

For each frozen predict-ready Cox: also reports mean-predicted S vs CG under
each τ (how the KM-calibration lens would shift under dependence).

Writes ``results/ladder/d0{n}_dependent_censoring.json``.

Execute:
    python -W default E04_rung4_dependent_censoring.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import cfg
from src.metrics.dependent_censoring import copula_sensitivity_sweep
from src.metrics.freeze import load_frozen_manifest
from src.metrics.io import utc_now, write_json
from src.metrics.predict import load_eval_frame, predict_survival_curves

warnings.filterwarnings("default")


def log(msg: str = "") -> None:
    print(msg, flush=True)


def _slug(domain_id: str) -> str:
    return domain_id.lower().replace("domain_", "d")


def _eval_times_events(artifact: dict) -> tuple | None:
    """Observed (time, event) on the eval frame (even without estimator curves)."""
    try:
        df = load_eval_frame(artifact)
    except Exception:  # noqa: BLE001
        return None
    tcol, ecol = artifact.get("time_col"), artifact.get("event_col")
    if not tcol or not ecol or tcol not in df.columns or ecol not in df.columns:
        return None
    work = df.dropna(subset=[tcol, ecol])
    work = work.loc[work[tcol] > 0]
    if len(work) == 0:
        return None
    return work[tcol].to_numpy(float), work[ecol].to_numpy(int)


def eval_one_model(artifact: dict) -> dict:
    mid = artifact["model_id"]
    base = {
        "model_id": mid,
        "h1_rank_key": artifact.get("h1_rank_key"),
        "sha256": artifact.get("sha256"),
        "eval_mode": artifact.get("eval_mode"),
    }

    if artifact.get("kind") == "cv_scores":
        # Still can sweep on eval labels if present
        te = _eval_times_events(artifact)
        if te is None:
            return {
                **base,
                "implementation_status": "skipped_no_estimator",
                "note": artifact.get("notes"),
            }
        sweep = copula_sensitivity_sweep(te[0], te[1])
        return {
            **base,
            "implementation_status": "live_population_only",
            "note": "No estimator curves — CopulaGraphic vs KM on eval labels only.",
            "sweep": sweep,
        }

    if not artifact.get("present"):
        return {**base, "implementation_status": "skipped_absent"}

    backend = (artifact.get("backend") or "").lower()
    can_curves = (
        artifact.get("predict_ready")
        and artifact.get("load_ok")
        and "xgb" not in backend
        and mid != "cox_xgboost"
    )

    try:
        if can_curves:
            curves = predict_survival_curves(artifact, n_grid=100)
            sweep = copula_sensitivity_sweep(
                curves["time"],
                curves["event"],
                surv_grid=curves["surv_grid"],
                times_grid=curves["times_grid"],
            )
            return {
                **base,
                "implementation_status": "live",
                "n": curves["n"],
                "n_events": curves["n_events"],
                "curve_backend": curves["curve_backend"],
                "metric_backend": "SurvivalEVAL.CopulaGraphic",
                "sweep": sweep,
            }

        te = _eval_times_events(artifact)
        if te is None:
            return {
                **base,
                "implementation_status": "skipped_no_survival_curve",
                "note": "No curves and no eval labels for CG sweep.",
            }
        sweep = copula_sensitivity_sweep(te[0], te[1])
        return {
            **base,
            "implementation_status": "live_population_only",
            "note": "Risk-only / no baseline — CG vs KM on eval labels only.",
            "metric_backend": "SurvivalEVAL.CopulaGraphic",
            "sweep": sweep,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            **base,
            "implementation_status": "error",
            "error": str(exc),
        }


def main() -> int:
    log("─" * 60)
    log("E04 — RUNG 4 DEPENDENT CENSORING (SurvivalEVAL CopulaGraphic)")
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
        n_live = sum(
            1
            for m in models
            if str(m.get("implementation_status", "")).startswith("live")
        )
        payload = {
            "stage": "E04",
            "rung": 4,
            "rung_name": "dependent_censoring",
            "anchor_parallel": "double-helix ladder rung 4 — dependent censoring",
            "generated_at_utc": utc_now(),
            "domain_id": domain_id,
            "implementation_status": "live" if n_live else "partial",
            "n_live_models": n_live,
            "models": models,
            "metric_backend": "SurvivalEVAL.CopulaGraphic",
            "limitation": "true_dependence_non_identifiable",
            "rule": (
                "No retraining. Clayton τ grid = anchor scenarios. "
                "Sweep is sensitivity under assumed dependence, not an estimate of τ."
            ),
        }
        path = cfg.DIRS["ladder"] / f"{_slug(domain_id)}_dependent_censoring.json"
        write_json(path, payload)
        out_paths.append(path)
        log(f"  {domain_id}: liveish_models={n_live}")
        for m in models:
            st = m.get("implementation_status")
            sw = m.get("sweep") or {}
            mae = sw.get("mae_cg_vs_km_by_tau") or {}
            if mae:
                bits = " ".join(f"τ={k}:MAE={v:.4f}" for k, v in mae.items() if v is not None)
                log(f"    {m['model_id']}: {st}  CG-vs-KM[{bits}]")
            else:
                log(f"    {m['model_id']}: {st}")

    log(f"E04 complete — {len(out_paths)} domain file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
