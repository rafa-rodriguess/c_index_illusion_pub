"""
D03b_DOMAIN_02_export_xgb_breslow.py — Attach Breslow S(t) to frozen XGB-Cox
===========================================================================
Does **not** retrain the Booster. Fits Breslow H₀ on the **train** split
using frozen XGB risk scores, then re-wraps the joblib as:

  {
    "model": Booster,
    "features": [...],
    "breslow": {event_times, H0, eta_train_mean, ...},
    "curve_backend": "xgb_cox_breslow",
  }

Also writes ``results/models/domain2/xgb_breslow_meta.json``.

Execute:
    python -W default D03b_DOMAIN_02_export_xgb_breslow.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import cfg
from src.metrics.io import write_json
from src.metrics.xgb_breslow import fit_breslow_bundle, predict_survival_breslow


def log(msg: str = "") -> None:
    print(msg, flush=True)


def main() -> int:
    log("─" * 60)
    log("D03b — EXPORT XGB-COX BRESLOW BASELINE (no retrain)")
    log("─" * 60)

    bst_path = cfg.DIRS["models"] / "domain2" / "xgb_cox.joblib"
    feat_path = cfg.ROOT / "data/processed/domain2/design_features.json"
    split_path = cfg.ROOT / "data/processed/domain2/loans_split.parquet"
    if not bst_path.exists():
        log(f"ERROR: missing {bst_path}")
        return 1
    if not split_path.exists():
        log(f"ERROR: missing {split_path}")
        return 1

    blob = joblib.load(bst_path)
    if isinstance(blob, dict) and "model" in blob:
        bst = blob["model"]
        feats = list(blob.get("features") or [])
        existing_breslow = blob.get("breslow")
    else:
        bst = blob
        feats = []
        existing_breslow = None

    if not feats and feat_path.exists():
        feats = list(json.loads(feat_path.read_text(encoding="utf-8")).get("features") or [])
    if not feats:
        log("ERROR: no feature list")
        return 1

    df = pd.read_parquet(split_path)
    tr = df.loc[df["split_role"] == "train"].copy()
    te = df.loc[df["split_role"] == "test"].copy()
    tcol = "duration_days_split" if "duration_days_split" in tr.columns else "duration_days"
    ecol = "event_split" if "event_split" in tr.columns else "event"
    tr = tr.dropna(subset=[tcol, ecol] + feats)
    tr = tr.loc[tr[tcol] > 0].copy()
    te = te.dropna(subset=[tcol, ecol] + feats)
    te = te.loc[te[tcol] > 0].copy()

    X_tr = tr[feats].fillna(0.0).to_numpy(dtype=np.float32)
    X_te = te[feats].fillna(0.0).to_numpy(dtype=np.float32)
    dtr = xgb.DMatrix(X_tr, feature_names=feats)
    dte = xgb.DMatrix(X_te, feature_names=feats)
    eta_tr = np.asarray(bst.predict(dtr), dtype=float)
    eta_te = np.asarray(bst.predict(dte), dtype=float)

    bundle = fit_breslow_bundle(
        tr[tcol].to_numpy(dtype=float),
        tr[ecol].to_numpy(dtype=int),
        eta_tr,
    )
    log(
        f"  Train n={bundle['n_train']:,} events={bundle['n_events_train']:,}  "
        f"Breslow knots={len(bundle['event_times']):,}"
    )

    # Smoke: S(t) on test at a few horizons
    grid = np.linspace(0.0, float(np.quantile(te[tcol], 0.9)), 50)
    grid[0] = 0.0
    pred = predict_survival_breslow(bundle, eta_te, grid, time_obs=te[tcol].to_numpy())
    S = pred["surv_grid"]
    log(
        f"  Test n={len(te):,}  S(grid) range=[{S.min():.4f},{S.max():.4f}]  "
        f"mean S@mid={float(S[len(S)//2].mean()):.4f}"
    )
    if not np.isfinite(S).all() or S.min() < 0 or S.max() > 1.0 + 1e-6:
        log("ERROR: invalid survival curves")
        return 1

    out_blob = {
        "model": bst,
        "features": feats,
        "breslow": {
            "method": bundle["method"],
            "eta_train_mean": bundle["eta_train_mean"],
            "event_times": bundle["event_times"].tolist(),
            "H0": bundle["H0"].tolist(),
            "n_train": bundle["n_train"],
            "n_events_train": bundle["n_events_train"],
        },
        "curve_backend": "xgb_cox_breslow",
        "time_col_train": tcol,
        "event_col_train": ecol,
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Breslow H0 fit on train split with frozen XGB scores; "
            "Booster weights unchanged (no retrain)."
        ),
    }
    # Backup raw booster once
    bak = cfg.DIRS["models"] / "domain2" / "xgb_cox_booster_only.joblib"
    if not bak.exists():
        joblib.dump(bst, bak)
        log(f"  Backup booster → {bak.relative_to(cfg.ROOT)}")

    joblib.dump(out_blob, bst_path)
    meta = {
        "stage": "D03b",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "path": str(bst_path.relative_to(cfg.ROOT)),
        "n_features": len(feats),
        "breslow_knots": len(bundle["event_times"]),
        "n_train": bundle["n_train"],
        "n_test_smoke": int(len(te)),
        "test_S_min": float(S.min()),
        "test_S_max": float(S.max()),
        "test_S_mean_mid": float(S[len(S) // 2].mean()),
        "replaced_prior_breslow": existing_breslow is not None,
    }
    write_json(cfg.DIRS["models"] / "domain2" / "xgb_breslow_meta.json", meta)
    log(f"  Wrote {bst_path.relative_to(cfg.ROOT)}")
    log(f"  Wrote results/models/domain2/xgb_breslow_meta.json")
    log("D03b complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
