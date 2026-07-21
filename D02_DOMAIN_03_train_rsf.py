"""
D02_DOMAIN_03_train_rsf.py — RSF 5-fold × 30 runs (Table 8 grid)
===============================================================
Trains Random Survival Forests for each
  community × feature set × θ × run × fold

Backend: PySurvival ``RandomSurvivalForestModel`` + ``concordance_index``
(author notebook / paper). Hyperparameters from the notebook:

  num_trees=5, max_features=sqrt, max_depth=5, min_node_size=30,
  sample_size_pct=0.63, importance_mode=permutation

CV (author notebook):
  train_test_split(test_size=0.01) then KFold(5) without shuffle on remainder.

Requires conda env ``d3-pysurvival`` (patched pysurvival 0.1.2). See
``domain3-abedi-2022/CODE_ACCESS.md``.

Execute:
    /opt/anaconda3/envs/d3-pysurvival/bin/python -W default D02_DOMAIN_03_train_rsf.py
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import cfg

warnings.filterwarnings("default")

try:
    from pysurvival.models.survival_forest import RandomSurvivalForestModel
    from pysurvival.utils.metrics import concordance_index
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "PySurvival is required for DOMAIN_03 D02.\n"
        "Use: /opt/anaconda3/envs/d3-pysurvival/bin/python -W default "
        "D02_DOMAIN_03_train_rsf.py\n"
        f"Import error: {exc}"
    ) from exc


def log(msg: str = "") -> None:
    print(msg, flush=True)


def feature_cols(name: str) -> list[str]:
    fs = cfg.DOMAIN_03["feature_sets"]
    if name == "combined":
        return list(fs["behavioural"]) + list(fs["content"])
    return list(fs[name])


def iter_folds(
    n: int,
    n_folds: int,
    n_runs: int,
    base_seed: int,
    holdout_frac: float,
    kfold_shuffle: bool,
):
    """Yield (run_id, fold_id, train_idx, eval_idx) on the post-holdout pool."""
    idx = np.arange(n)
    for run_id in range(n_runs):
        seed = int(base_seed) + run_id
        pool, _holdout = train_test_split(
            idx,
            test_size=holdout_frac,
            random_state=seed,
        )
        pool = np.asarray(pool)
        kf = KFold(n_splits=n_folds, shuffle=kfold_shuffle)
        for fold_id, (tr_rel, te_rel) in enumerate(kf.split(pool)):
            yield run_id, fold_id, pool[tr_rel], pool[te_rel]


def fit_rsf(X_train, T_train, E_train, rsf_cfg: dict, seed: int | None):
    model = RandomSurvivalForestModel(num_trees=int(rsf_cfg["n_estimators"]))
    model.fit(
        X_train,
        T_train,
        E_train,
        max_features=rsf_cfg["max_features"],
        max_depth=int(rsf_cfg["max_depth"]),
        min_node_size=int(rsf_cfg["min_samples_leaf"]),
        num_threads=int(rsf_cfg.get("n_jobs", -1)),
        weights=None,
        sample_size_pct=float(rsf_cfg.get("sample_size_pct", 0.63)),
        importance_mode=rsf_cfg.get("importance_mode", "permutation"),
        seed=seed,
        save_memory=False,
    )
    return model


def main() -> int:
    log("─" * 60)
    log("D02_DOMAIN_03 — TRAIN RSF (Table 8 grid, PySurvival)")
    log("─" * 60)

    cv_policy = cfg.DIRS["processed_d3"] / "cv_policy.json"
    if not cv_policy.exists():
        log("ERROR: run D01 first")
        return 1

    rsf_cfg = cfg.DOMAIN_03["rsf"]
    cv = cfg.DOMAIN_03["cv"]
    n_folds = int(cv["n_folds"])
    n_runs = int(cv["n_runs"])
    holdout_frac = float(cv.get("holdout_frac", 0.01))
    kfold_shuffle = bool(cv.get("kfold_shuffle", False))
    feature_set_names = ("behavioural", "content", "combined")

    out_dir = cfg.DIRS["models"] / "domain3"
    out_dir.mkdir(parents=True, exist_ok=True)

    cells: dict[str, dict] = {}
    all_scores: list[dict] = []

    total = (
        len(cfg.DOMAIN_03["communities"])
        * len(feature_set_names)
        * len(cfg.DOMAIN_03["theta_months"])
    )
    done = 0

    for code in cfg.DOMAIN_03["communities"]:
        for theta in cfg.DOMAIN_03["theta_months"]:
            df = pd.read_parquet(cfg.DIRS["processed_d3"] / f"{code}_theta{theta}.parquet")
            n = len(df)
            for fset in feature_set_names:
                cols = feature_cols(fset)
                X_all = (
                    df[cols]
                    .apply(pd.to_numeric, errors="coerce")
                    .fillna(0.0)
                    .astype(np.float64)
                )
                T_all = df["duration_months"].to_numpy(dtype=np.float64)
                E_all = df["event"].to_numpy(dtype=np.int32)
                scores: list[float] = []

                for run_id, fold_id, tr, te in iter_folds(
                    n,
                    n_folds,
                    n_runs,
                    cfg.RANDOM_SEED,
                    holdout_frac,
                    kfold_shuffle,
                ):
                    seed = int(cfg.RANDOM_SEED) + 1000 * run_id + fold_id
                    model = fit_rsf(
                        X_all.iloc[tr],
                        T_all[tr],
                        E_all[tr],
                        rsf_cfg,
                        seed=seed,
                    )
                    c = float(
                        concordance_index(
                            model,
                            X_all.iloc[te],
                            T_all[te],
                            E_all[te],
                        )
                    )
                    scores.append(c)
                    all_scores.append(
                        {
                            "community": code,
                            "feature_set": fset,
                            "theta": int(theta),
                            "run": run_id,
                            "fold": fold_id,
                            "cindex": c,
                        }
                    )

                key = f"{code}|{fset}|{theta}"
                mean_c = float(np.mean(scores))
                std_c = float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0
                paper = cfg.DOMAIN_03["paper_table8"].get((code, fset, int(theta)))
                author_bin = None
                cells[key] = {
                    "community": code,
                    "community_label": cfg.DOMAIN_03["community_labels"][code],
                    "feature_set": fset,
                    "theta": int(theta),
                    "n_scores": len(scores),
                    "cindex_mean": mean_c,
                    "cindex_std": std_c,
                    "paper_cindex_mean": paper,
                    "gap": (mean_c - paper) if paper is not None else None,
                    "author_bin_mean": author_bin,
                }
                done += 1
                gap_s = f"{(mean_c - paper):+.3f}" if paper is not None else "—"
                log(
                    f"  [{done}/{total}] {code} {fset} θ={theta}: "
                    f"C={mean_c:.3f}±{std_c:.3f}  paper={paper}  gap={gap_s}"
                )

    metrics = {
        "stage": "D02_DOMAIN_03",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": rsf_cfg["backend"],
        "author_backend": rsf_cfg["author_backend"],
        "rsf_hyperparameters": {
            "n_estimators": rsf_cfg["n_estimators"],
            "max_depth": rsf_cfg["max_depth"],
            "min_samples_leaf": rsf_cfg["min_samples_leaf"],
            "max_features": rsf_cfg["max_features"],
            "sample_size_pct": rsf_cfg.get("sample_size_pct", 0.63),
            "importance_mode": rsf_cfg.get("importance_mode", "permutation"),
        },
        "cv": {
            "n_folds": n_folds,
            "n_runs": n_runs,
            "holdout_frac": holdout_frac,
            "kfold_shuffle": kfold_shuffle,
        },
        "contributor_filter": bool(cfg.DOMAIN_03.get("contributor_filter", True)),
        "cells": cells,
        "protocol_deviations": [
            "PySurvival built from 0.1.2 source with tp_print patched for Python 3.9/macOS",
            "holdout seed = RANDOM_SEED+run_id (author notebook used seed=None)",
            "Contributor filter Q∪A∪C∪U∪D applied (paper §5.2; absent from public notebook)",
        ],
    }
    (out_dir / "rsf_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pd.DataFrame(all_scores).to_csv(out_dir / "rsf_fold_scores.csv", index=False)

    log(f"  Wrote {out_dir.relative_to(cfg.ROOT)}/rsf_metrics.json")
    log("D02_DOMAIN_03 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
