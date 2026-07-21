"""
D02_DOMAIN_01_train_cox.py — Fit Ahmed & Green Cox PH (DOMAIN_01)
=================================================================
Fits ``lifelines.CoxPHFitter`` on the H6a DOMAIN_01 cohort and reports
Harrell concordance (target 0.958).

Paper evidence:
  - §6 names lifelines.CoxPHFitter
  - §7.1 reports c-index 0.958 as goodness-of-fit; highlights HR for
    SMART 184 / 190 / 194
  - §4.1 sample counts (12,993 healthy + 4,889 failed) interpreted as
    informative sampling: healthy calendar-span >7y UNION all failed (H6a).
    Smoke 2026-07-12: in-sample C≈0.9595 on that population.
  - SMART 190 and 194 are identical on this Seagate model → drop 194
    (cfg.DOMAIN_01['cox_drop_collinear_smart']) for convergence
  - Small L2 ``cox_penalizer`` (default 0.01) for Newton convergence;
    paper silent → logged as deviation
  - Author GitLab URL 404 (2026-07-12)

Execute:
    python -W default D02_DOMAIN_01_train_cox.py
"""

from __future__ import annotations

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
from lifelines.utils import concordance_index

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import cfg
from src.domain1_cox_cohort import select_cox_fit_rows

warnings.filterwarnings("default")


def log(msg: str = "") -> None:
    print(msg, flush=True)


def load_drives() -> pd.DataFrame:
    d = cfg.DIRS["processed_d1"]
    for name in ("drives.parquet", "drives.csv.gz"):
        p = d / name
        if p.exists():
            return pd.read_parquet(p) if name.endswith(".parquet") else pd.read_csv(p)
    raise FileNotFoundError("drives table missing — run D00.")


def main() -> int:
    log("─" * 60)
    log("D02_DOMAIN_01 — TRAIN COX (Ahmed & Green)")
    log("─" * 60)

    smart_cols = [f"smart_{sid}_raw" for sid in cfg.DOMAIN_01["smart_ids"]]
    drop_ids = list(cfg.DOMAIN_01.get("cox_drop_collinear_smart") or [])
    drop_cols = [f"smart_{sid}_raw" for sid in drop_ids]
    fit_cols = [c for c in smart_cols if c not in drop_cols]
    penalizer = float(cfg.DOMAIN_01.get("cox_penalizer", 0.01))
    pop = cfg.DOMAIN_01["cox_fit_population"]

    df = load_drives()
    cohort = select_cox_fit_rows(df)
    log(f"  Drives loaded: {len(df):,}")
    log(f"  Fit population: {pop}")
    log(
        f"  Cohort rows : {len(cohort):,} "
        f"(healthy={(cohort['event']==0).sum():,} failed={(cohort['event']==1).sum():,})"
    )
    log(f"  Fit cols     : {len(fit_cols)} SMART (dropped collinear: {drop_ids})")
    log(f"  Penalizer    : {penalizer}")

    use_cols = ["duration_days", "event"] + fit_cols
    data = cohort[use_cols].copy()
    n_before = len(data)
    data = data.dropna()
    data = data.loc[data["duration_days"] > 0].copy()
    n_drop = n_before - len(data)
    if n_drop:
        log(f"  Dropped {n_drop:,} incomplete / non-positive duration rows")

    if len(data) < 100 or int(data["event"].sum()) < 10:
        log("ERROR: too few complete cases / events for Cox fit.")
        return 1

    cph = CoxPHFitter(penalizer=penalizer)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cph.fit(
                data,
                duration_col="duration_days",
                event_col="event",
                show_progress=False,
            )
        for w in caught:
            log(f"  Warning: {w.category.__name__}: {w.message}")
    except ConvergenceError as exc:
        log(f"ERROR: Cox failed to converge: {exc}")
        return 1

    risk = cph.predict_partial_hazard(data).values.ravel()
    c_index = float(
        concordance_index(
            data["duration_days"].values,
            -risk,
            data["event"].values,
        )
    )
    c_concordance_ = float(cph.concordance_index_)

    out_dir = cfg.DIRS["models"] / "domain1"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "cox_ahmed_green.joblib"
    joblib.dump(
        {
            "model": cph,
            "smart_cols": fit_cols,
            "dropped_cols": drop_cols,
            "penalizer": penalizer,
            "fit_n": len(data),
            "fit_population": pop,
            "protocol": "DOMAIN_01",
        },
        model_path,
    )

    summary = cph.summary.copy()
    hr_table = []
    for cov in fit_cols:
        if cov not in summary.index:
            continue
        row = summary.loc[cov]
        hr_table.append(
            {
                "covariate": cov,
                "coef": float(row["coef"]),
                "exp_coef": float(row["exp(coef)"]),
                "p": float(row["p"]),
            }
        )

    highlights = {}
    for sid, paper_hr in [(184, 1.010), (190, 0.984), (194, 0.990)]:
        col = f"smart_{sid}_raw"
        if col in cph.params_.index:
            highlights[col] = {
                "ours_hr": float(np.exp(cph.params_[col])),
                "paper_hr": paper_hr,
            }
        elif sid in drop_ids:
            highlights[col] = {
                "ours_hr": None,
                "paper_hr": paper_hr,
                "note": "dropped as collinear duplicate (cfg)",
            }

    deviations = [
        (
            f"Author code URL {cfg.DOMAIN_01.get('author_code_url')} status="
            f"{cfg.DOMAIN_01.get('author_code_status')} — cannot verify exact lifelines kwargs"
        ),
        "Covariates = last-day SMART raw (paper underspecified)",
        (
            "Cox fit population H6a: healthy calendar_span_years>7 UNION all failed "
            "(paper §4.1 counts; smoke recovers C≈0.958). Not SMART9-both-classes."
        ),
        f"Dropped collinear SMART ids {drop_ids} (190≡194 on this model)",
        f"Used cox_penalizer={penalizer} for Newton convergence (paper silent)",
        "In-sample GOF (paper does not describe Cox hold-out; 80/20 is DeepNet)",
    ]

    metrics = {
        "stage": "D02_DOMAIN_01",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": cfg.DOMAIN_01["cox_backend"],
        "n_fit": len(data),
        "n_events": int(data["event"].sum()),
        "n_dropped_nan": int(n_drop),
        "smart_cols_fit": fit_cols,
        "dropped_collinear": drop_cols,
        "penalizer": penalizer,
        "harrell_cindex": c_index,
        "lifelines_concordance_index_": c_concordance_,
        "target_cindex": cfg.DOMAIN_01["target_value"],
        "delta_vs_target": c_index - float(cfg.DOMAIN_01["target_value"]),
        "eval_mode": "in_sample_gof",
        "fit_population": pop,
        "model_path": str(model_path.relative_to(cfg.ROOT)),
        "hazard_ratios": hr_table,
        "paper_hr_highlights": highlights,
        "protocol_deviations": deviations,
    }
    metrics_path = out_dir / "cox_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    cph.summary.to_csv(out_dir / "cox_summary.csv")

    log(f"  Fitted n={len(data):,}  events={int(data['event'].sum()):,}")
    log(f"  Harrell C = {c_index:.4f}  (target {cfg.DOMAIN_01['target_value']})")
    log(f"  Δ         = {metrics['delta_vs_target']:+.4f}")
    for col, info in highlights.items():
        log(f"  {col}: ours_HR={info.get('ours_hr')}  paper_HR={info.get('paper_hr')}")
    log(f"  Wrote     {model_path.relative_to(cfg.ROOT)}")
    log(f"  Wrote     {metrics_path.relative_to(cfg.ROOT)}")
    log("D02_DOMAIN_01 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
