"""
D02b_DOMAIN_03_export_ladder_rsf.py — Freeze RSF estimators for Block E
=======================================================================
Fits sksurv RSF on Politics θ=24 panels (current processed D3) for the three
H1 ranking feature sets, using the author CV split rule (1% holdout + KFold
fold-0 of run-0 as ladder eval).

Does **not** replace Table-8 CV scores (``rsf_fold_scores`` / ``rsf_metrics``).
Writes predict-ready joblibs for E00–E06.

Execute:
    python -W default D02b_DOMAIN_03_export_ladder_rsf.py
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
from sklearn.model_selection import KFold, train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import cfg

warnings.filterwarnings("default")


def log(msg: str = "") -> None:
    print(msg, flush=True)


def feature_cols(name: str) -> list[str]:
    fs = cfg.DOMAIN_03["feature_sets"]
    if name == "combined":
        return list(fs["behavioural"]) + list(fs["content"])
    return list(fs[name])


def main() -> int:
    log("─" * 60)
    log("D02b — EXPORT D3 LADDER RSF (P / θ=24)")
    log("─" * 60)

    panel = cfg.DIRS["processed_d3"] / "p_theta24.parquet"
    if not panel.exists():
        log(f"ERROR: missing {panel}")
        return 1

    df = pd.read_parquet(panel)
    time_col, event_col = "duration_months", "event"
    rsf_cfg = cfg.DOMAIN_03["rsf"]
    cv = cfg.DOMAIN_03["cv"]
    seed = int(cfg.RANDOM_SEED)

    idx = np.arange(len(df))
    pool, _hold = train_test_split(
        idx,
        test_size=float(cv["holdout_frac"]),
        random_state=seed,  # run_id=0
    )
    pool = np.asarray(pool)
    tr_rel, te_rel = next(
        KFold(n_splits=int(cv["n_folds"]), shuffle=bool(cv["kfold_shuffle"])).split(pool)
    )
    train_idx, eval_idx = pool[tr_rel], pool[te_rel]
    eval_df = df.iloc[eval_idx].copy()
    out_dir = cfg.DIRS["models"] / "domain3"
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_path = out_dir / "ladder_eval_p_theta24.parquet"
    eval_df.to_parquet(eval_path, index=False)
    log(f"  Eval fold: n={len(eval_df)} → {eval_path.relative_to(cfg.ROOT)}")

    y_train = Surv.from_arrays(
        df.iloc[train_idx][event_col].to_numpy(dtype=bool),
        df.iloc[train_idx][time_col].to_numpy(dtype=float),
    )

    meta = {
        "stage": "D02b_DOMAIN_03",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "community": "p",
        "theta": 24,
        "time_col": time_col,
        "event_col": event_col,
        "split": {
            "rule": "holdout 1% (seed=RANDOM_SEED) then KFold fold-0 train/eval",
            "n_train": int(len(train_idx)),
            "n_eval": int(len(eval_idx)),
            "random_seed": seed,
        },
        "rsf": {
            "backend": "sksurv.ensemble.RandomSurvivalForest",
            "n_estimators": int(rsf_cfg["n_estimators"]),
            "max_depth": int(rsf_cfg["max_depth"]),
            "min_samples_leaf": int(rsf_cfg["min_samples_leaf"]),
            "max_features": rsf_cfg["max_features"],
        },
        "models": {},
    }

    for fset in ("behavioural", "content", "combined"):
        cols = feature_cols(fset)
        X_tr = df.iloc[train_idx][cols].fillna(0.0).to_numpy(dtype=float)
        X_te = df.iloc[eval_idx][cols].fillna(0.0).to_numpy(dtype=float)
        model = RandomSurvivalForest(
            n_estimators=int(rsf_cfg["n_estimators"]),
            max_depth=int(rsf_cfg["max_depth"]),
            min_samples_leaf=int(rsf_cfg["min_samples_leaf"]),
            max_features=rsf_cfg["max_features"],
            n_jobs=-1,
            random_state=seed,
        )
        model.fit(X_tr, y_train)
        risk = model.predict(X_te)
        from sksurv.metrics import concordance_index_censored

        c = concordance_index_censored(
            eval_df[event_col].to_numpy(bool),
            eval_df[time_col].to_numpy(float),
            risk,
        )[0]
        blob = {
            "model": model,
            "features": cols,
            "feature_set": fset,
            "community": "p",
            "theta": 24,
            "time_col": time_col,
            "event_col": event_col,
            "backend": "sksurv.ensemble.RandomSurvivalForest",
            "ladder_eval_path": str(eval_path.relative_to(cfg.ROOT)),
            "harrell_eval": float(c),
            "protocol": meta["split"],
        }
        mid = f"rsf_{fset}"
        path = out_dir / f"{mid}_p_theta24.joblib"
        joblib.dump(blob, path)
        meta["models"][mid] = {
            "path": str(path.relative_to(cfg.ROOT)),
            "n_features": len(cols),
            "harrell_eval": float(c),
            "features": cols,
        }
        log(f"  {mid}: Harrell_eval={c:.4f} → {path.relative_to(cfg.ROOT)}")

    meta_path = out_dir / "ladder_rsf_export.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    log(f"  Wrote {meta_path.relative_to(cfg.ROOT)}")
    log("D02b complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
