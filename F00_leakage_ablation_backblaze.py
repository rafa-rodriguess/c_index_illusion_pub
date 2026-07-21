"""
F00_leakage_ablation_backblaze.py — Companion ablated Cox for H1 D1 ranking
==========================================================================
Fits Cox **without** SMART ids in ``cfg.H4_ABLATION_SMART``, compares to the
frozen full Cox on the same DOMAIN_01 eval population, and writes
``cox_ablated_h4.joblib`` for the ladder / H1 ranking object ``cox_ablated``.

This is **not** the H4 hypothesis exhibit. H4 primary = leave-one-out survey
(``F00_sens1``) + paired bootstrap (``H04``).

Execute:
    python -W default F00_leakage_ablation_backblaze.py
"""

from __future__ import annotations

import csv
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
from sksurv.metrics import concordance_index_censored

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import cfg
from src.metrics.calibration import d_calibration, make_survival_evaluator
from src.metrics.io import write_json
from src.metrics.predict import predict_survival_curves

warnings.filterwarnings("default")


def log(msg: str = "") -> None:
    print(msg, flush=True)


def _smart_col(sid: int) -> str:
    return f"smart_{int(sid)}_raw"


def _harrell(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    return float(
        concordance_index_censored(
            event.astype(bool), time.astype(float), risk.astype(float)
        )[0]
    )


def _fit_cox(data: pd.DataFrame, fit_cols: list[str], penalizer: float) -> CoxPHFitter:
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(
        data[["duration_days", "event"] + fit_cols],
        duration_col="duration_days",
        event_col="event",
        show_progress=False,
    )
    return cph


def main() -> int:
    log("─" * 60)
    log("F00 — COMPANION ABLATED COX (H1 D1 ranking; not H4 exhibit)")
    log("─" * 60)

    ablate_ids = list(cfg.H4_ABLATION_SMART)
    ablate_cols = [_smart_col(s) for s in ablate_ids]
    delta_thr = float(cfg.PROTOCOL["hypotheses"]["H4"]["delta_c_threshold"])
    # Full protocol B=1000 is expensive on ~37k rows; default 200 with --full for 1000.
    B = 1000 if "--full" in sys.argv else min(200, int(cfg.EVAL["n_bootstrap"]))
    seed = int(cfg.RANDOM_SEED)
    penalizer = float(cfg.DOMAIN_01.get("cox_penalizer", 0.01))

    full_path = cfg.DIRS["models"] / "domain1" / "cox_ahmed_green.joblib"
    if not full_path.exists():
        log("ERROR: missing full Cox — run D02_DOMAIN_01_train_cox.py")
        return 1
    full_blob = joblib.load(full_path)
    full_model: CoxPHFitter = full_blob["model"]
    full_cols = list(full_blob["smart_cols"])
    ablated_cols = [c for c in full_cols if c not in ablate_cols]
    if len(ablated_cols) == len(full_cols):
        log(f"ERROR: ablation cols {ablate_cols} not in full feature set")
        return 1

    drives = cfg.DIRS["processed_d1"] / "drives.parquet"
    df = pd.read_parquet(drives)
    from src.domain1_cox_cohort import select_cox_fit_rows

    cohort = select_cox_fit_rows(df)
    use = cohort[["duration_days", "event"] + full_cols].dropna()
    use = use.loc[use["duration_days"] > 0].copy()
    time = use["duration_days"].to_numpy(float)
    event = use["event"].to_numpy(int)
    log(f"  n={len(use):,}  events={int(event.sum()):,}  pop={cfg.DOMAIN_01['cox_fit_population']}")
    log(f"  Ablate SMART ids: {ablate_ids} → drop {ablate_cols}")
    log(f"  Features: full={len(full_cols)}  ablated={len(ablated_cols)}")

    # Fit ablated on same rows as Fase A GOF
    try:
        ablated = _fit_cox(use, ablated_cols, penalizer)
    except ConvergenceError as exc:
        log(f"ERROR: ablated Cox failed: {exc}")
        return 1

    risk_full = np.asarray(full_model.predict_partial_hazard(use[full_cols]), float).ravel()
    risk_ab = np.asarray(ablated.predict_partial_hazard(use[ablated_cols]), float).ravel()
    c_full = _harrell(time, event, risk_full)
    c_ab = _harrell(time, event, risk_ab)
    delta_c = c_full - c_ab
    log(f"  Harrell full={c_full:.4f}  ablated={c_ab:.4f}  Δ={delta_c:.4f}")

    # Paired bootstrap on ΔC (subsample for speed; full n via --full)
    log(f"  Bootstrap B={B} (pass --full for B=1000)…")
    rng = np.random.default_rng(seed)
    n = len(use)
    boot_n = n if "--full" in sys.argv else min(8_000, n)
    boots_full = np.empty(B)
    boots_ab = np.empty(B)
    boots_delta = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=boot_n)
        bf = _harrell(time[idx], event[idx], risk_full[idx])
        ba = _harrell(time[idx], event[idx], risk_ab[idx])
        boots_full[b] = bf
        boots_ab[b] = ba
        boots_delta[b] = bf - ba
        if (b + 1) % 50 == 0:
            log(f"    … {b + 1}/{B}")
    lo_f, hi_f = np.quantile(boots_full, [0.025, 0.975])
    lo_a, hi_a = np.quantile(boots_ab, [0.025, 0.975])
    lo_d, hi_d = np.quantile(boots_delta, [0.025, 0.975])
    ci_nonoverlap = bool(lo_f > hi_a)  # full CI entirely above ablated CI
    delta_ge_thr = bool(delta_c >= delta_thr)
    # Protocol: drop > 0.03 with non-overlapping CIs
    h4_reject = bool(delta_ge_thr and ci_nonoverlap)
    log(
        f"  Bootstrap B={B}: Δ CI=[{lo_d:.4f},{hi_d:.4f}]  "
        f"CI_nonoverlap={ci_nonoverlap}  H4_preview={h4_reject}"
    )

    # Persist ablated model for Block E
    out_dir = cfg.DIRS["models"] / "domain1"
    out_dir.mkdir(parents=True, exist_ok=True)
    ab_path = out_dir / "cox_ablated_h4.joblib"
    joblib.dump(
        {
            "model": ablated,
            "smart_cols": ablated_cols,
            "ablated_smart_ids": ablate_ids,
            "ablated_cols": ablate_cols,
            "dropped_cols": list(full_blob.get("dropped_cols") or []) + ablate_cols,
            "penalizer": penalizer,
            "fit_n": len(use),
            "protocol": "F00_H4_ablation",
            "harrell_cindex": c_ab,
        },
        ab_path,
    )
    ab_metrics = {
        "stage": "F00",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "harrell_cindex": c_ab,
        "n_fit": len(use),
        "n_events": int(event.sum()),
        "smart_cols_fit": ablated_cols,
        "ablated_smart_ids": ablate_ids,
        "penalizer": penalizer,
        "compared_to_full": c_full,
        "delta_c_full_minus_ablated": delta_c,
    }
    ab_metrics_path = out_dir / "cox_ablated_h4_metrics.json"
    ab_metrics_path.write_text(json.dumps(ab_metrics, indent=2) + "\n", encoding="utf-8")
    log(f"  Wrote {ab_path.relative_to(cfg.ROOT)}")

    # D-Cal before/after (SurvivalEVAL) — reuse ladder predict path via fake artifacts
    dcal_full = dcal_ab = None
    try:
        art_full = {
            "kind": "estimator",
            "predict_ready": True,
            "load_ok": True,
            "path": str(full_path.relative_to(cfg.ROOT)),
            "eval_data_path": str(drives.relative_to(cfg.ROOT)),
            "time_col": "duration_days",
            "event_col": "event",
            "feature_cols": full_cols,
            "model_id": "cox_full",
            "domain_id": "DOMAIN_01",
            "backend": "lifelines.CoxPHFitter",
        }
        art_ab = {
            **art_full,
            "path": str(ab_path.relative_to(cfg.ROOT)),
            "feature_cols": ablated_cols,
            "model_id": "cox_ablated",
        }
        for label, art in (("full", art_full), ("ablated", art_ab)):
            curves = predict_survival_curves(art, n_grid=80)
            ev = make_survival_evaluator(
                curves["surv_grid"],
                curves["times_grid"],
                curves["time"],
                curves["event"],
                train_time=curves["time"],
                train_event=curves["event"],
            )
            dc = d_calibration(ev)
            if label == "full":
                dcal_full = dc
            else:
                dcal_ab = dc
            log(
                f"  D-Cal {label}: p={dc.get('p_value'):.3g} "
                f"reject={dc.get('reject_h0_well_calibrated')}"
            )
    except Exception as exc:  # noqa: BLE001
        log(f"  WARN: D-Cal compare failed: {exc}")

    # Paper probe artifact
    probe_dir = cfg.DIRS["probes"]
    probe_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "quantity_id": "delta_c_h4",
        "full_c": c_full,
        "ablated_c": c_ab,
        "delta_c": delta_c,
        "delta_threshold": delta_thr,
        "bootstrap_B": B,
        "bootstrap_n": boot_n,
        "ci_full": [float(lo_f), float(hi_f)],
        "ci_ablated": [float(lo_a), float(hi_a)],
        "ci_delta": [float(lo_d), float(hi_d)],
        "ci_nonoverlap": ci_nonoverlap,
        "h4_preview_reject": h4_reject,
        "role": "h1_d1_ranking_companion",
        "ablated_smart_ids": ablate_ids,
        "dcal_full_p": (dcal_full or {}).get("p_value"),
        "dcal_full_reject": (dcal_full or {}).get("reject_h0_well_calibrated"),
        "dcal_ablated_p": (dcal_ab or {}).get("p_value"),
        "dcal_ablated_reject": (dcal_ab or {}).get("reject_h0_well_calibrated"),
    }
    doc = {
        "stage": "F00",
        "probe": "companion_ablated_cox_h1_d1",
        "hypothesis": None,
        "note": (
            "Builds cox_ablated for H1 D1 ranking. Not an H4 exhibit — "
            "H4 = F00_sens1 LOO + H04 bootstrap."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "partial": False,
        "caption_draft": (
            f"Companion ablated Cox for H1 (D1 ranking object cox_ablated): "
            f"Harrell C {c_full:.3f} → {c_ab:.3f} (Δ={delta_c:.3f}). "
            "H4 decisions use the leave-one-out survey and H04, not this artifact."
        ),
        "result": row,
        "artifacts": {
            "full_model": str(full_path.relative_to(cfg.ROOT)),
            "ablated_model": str(ab_path.relative_to(cfg.ROOT)),
            "ablated_metrics": str(ab_metrics_path.relative_to(cfg.ROOT)),
        },
    }
    stem = probe_dir / "F00_h4_ablation"
    write_json(Path(str(stem) + ".json"), doc)

    md = [
        "# F00 — Companion ablated Cox (H1 D1 ranking)",
        "",
        f"_Generated UTC: `{doc['generated_at_utc']}`_",
        "",
        "> **Not an H4 exhibit.** H4 primary = `F00_sens1_leave_one_out` + `H04_paired_bootstrap_loo`.",
        "",
        f"- Dropped SMART ids (engineering drop for `cox_ablated`): `{ablate_ids}`",
        f"- Harrell **full** = `{c_full:.4f}` (CI [{lo_f:.4f}, {hi_f:.4f}])",
        f"- Harrell **ablated** = `{c_ab:.4f}` (CI [{lo_a:.4f}, {hi_a:.4f}])",
        f"- **ΔC** = `{delta_c:.4f}` (Δ CI [{lo_d:.4f}, {hi_d:.4f}])",
        f"- Non-overlapping CIs: `{ci_nonoverlap}`",
        "",
        "## Caption draft",
        "",
        doc["caption_draft"],
        "",
    ]
    Path(str(stem) + ".md").write_text("\n".join(md), encoding="utf-8")

    tex = [
        "% Auto-generated by F00_leakage_ablation_backblaze.py",
        "\\begin{tabular}{lrr}",
        "\\toprule",
        "Model & Harrell $C$ & 95\\% CI \\\\",
        "\\midrule",
        f"Full Cox & {c_full:.4f} & [{lo_f:.4f}, {hi_f:.4f}] \\\\",
        f"Ablated (no SMART {', '.join(map(str, ablate_ids))}) & {c_ab:.4f} & [{lo_a:.4f}, {hi_a:.4f}] \\\\",
        f"$\\Delta C$ & {delta_c:.4f} & [{lo_d:.4f}, {hi_d:.4f}] \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ]
    Path(str(stem) + ".tex").write_text("\n".join(tex), encoding="utf-8")

    with Path(str(stem) + ".csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)

    log(f"  Paper probe: {stem.relative_to(cfg.ROOT)}.*")
    log("F00 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
