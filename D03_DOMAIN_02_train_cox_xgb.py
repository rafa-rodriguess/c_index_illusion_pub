"""
D03_DOMAIN_02_train_cox_xgb.py — Cox + XGB-Cox (Optuna on validation)
=====================================================================
Paper §3.4–3.6:
  - lifelines CoxPHFitter on one-hot design (§3.2)
  - XGBoost survival:cox with Optuna TPE on **validation** set
  - Ratings = equal-mass HR bins on training predictions (7 = AA–F)
  - Table 1 defaults on Dömötör completed loans; IRR skipped

Execute:
    python -W default D03_DOMAIN_02_train_cox_xgb.py
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
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import cfg

warnings.filterwarnings("default")


def log(msg: str = "") -> None:
    print(msg, flush=True)


META = {
    "LoanId",
    "LoanDate",
    "LoanDuration",
    "duration_days",
    "duration_days_split",
    "event",
    "event_split",
    "early_repayment",
    "DefaultDate",
    "ContractEndDate",
    "LastPaymentOn",
    "Status",
    "Rating",
    "split_role",
    "obs_end",
    "is_completed",
}

# Bondora Status values treated as closed (paper Table 1 / footnote 14).
_CLOSED_STATUS = {"Repaid", "Closed", "Finished"}


def design_cols(df: pd.DataFrame) -> list[str]:
    feat_path = cfg.DIRS["processed_d2"] / "design_features.json"
    if feat_path.exists():
        names = json.loads(feat_path.read_text(encoding="utf-8"))["features"]
        return [c for c in names if c in df.columns]
    return [c for c in df.columns if c not in META]


def assign_ratings(scores: np.ndarray, edges: np.ndarray, labels: list[str]) -> np.ndarray:
    bins = np.digitize(scores, edges[1:-1], right=True)
    bins = np.clip(bins, 0, len(labels) - 1)
    return np.asarray(labels)[bins]


def mark_completed(
    df: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
    inactive_days: int | None = None,
) -> pd.Series:
    """
    Dömötör-style completed loans (Bone-Winkel Table 1 / footnote 14):

      completed = Bondora-closed OR no payment for ≥ inactive_days (default 365).

    Used as the Table 1 denominator for default rates (and IRR when repayments exist).
    """
    if inactive_days is None:
        inactive_days = int(cfg.DOMAIN_02.get("completed_inactive_days", 365))
    if as_of is None:
        as_of = pd.Timestamp(cfg.DOMAIN_02["paper_retrieve_date"])

    done = pd.Series(False, index=df.index)
    if "Status" in df.columns:
        done |= df["Status"].astype(str).isin(_CLOSED_STATUS)

    if "LastPaymentOn" in df.columns:
        last_pay = pd.to_datetime(df["LastPaymentOn"], errors="coerce")
        cutoff = as_of - pd.Timedelta(days=int(inactive_days))
        # Missing last payment → treat as inactive (conservative; matches "no payment")
        inactive = last_pay.isna() | (last_pay <= cutoff)
        done |= inactive
    elif "DefaultDate" in df.columns:
        # Fallback if LastPaymentOn absent: defaulted loans count as resolved outcomes
        done |= pd.to_datetime(df["DefaultDate"], errors="coerce").notna()

    return done


def rating_performance_table(
    df: pd.DataFrame,
    rating_col: str,
    term_days: float,
    completed: pd.Series | None = None,
) -> dict[str, dict]:
    """Empirical (all), KM@term, and Dömötör completed-only default rates per rating."""
    out: dict[str, dict] = {}
    if completed is None:
        completed = mark_completed(df)
    for r, g in df.groupby(rating_col, dropna=False):
        g = g.copy()
        n = len(g)
        emp = float(g["event"].mean()) if n else None
        km_def = None
        if n >= 5 and g["event"].sum() >= 0:
            try:
                km = KaplanMeierFitter()
                km.fit(g["duration_days"], g["event"], label=str(r))
                t = min(float(term_days), float(g["duration_days"].max()))
                surv = float(km.predict(t))
                km_def = float(1.0 - surv)
            except Exception:  # noqa: BLE001
                km_def = None
        gc = g.loc[completed.loc[g.index]]
        n_c = int(len(gc))
        emp_c = float(gc["event"].mean()) if n_c else None
        # Table 1 primary key = completed (Dömötör); keep empirical aliases
        out[str(r)] = {
            "n": int(n),
            "default_rate": emp_c if emp_c is not None else emp,
            "default_rate_empirical": emp,
            "default_rate_km_at_term": km_def,
            "n_completed": n_c,
            "default_rate_completed": emp_c,
            "mean_interest": float(g["Interest"].mean())
            if "Interest" in g.columns and n
            else None,
            "mean_interest_completed": float(gc["Interest"].mean())
            if "Interest" in g.columns and n_c
            else None,
        }
    return out


def make_cox_label(duration: np.ndarray, event: np.ndarray) -> np.ndarray:
    y = duration.astype(np.float32).copy()
    y[event == 0] *= -1.0
    return y


def _xgb_tree_method(prefer_gpu: bool) -> tuple[str, str]:
    """Return (tree_method, device_note). Prefer GPU as in paper §3.4; fall back to hist."""
    import xgboost as xgb

    if not prefer_gpu:
        return "hist", "CPU hist (GPU disabled in config)"
    try:
        d = xgb.DMatrix([[0.0, 1.0], [1.0, 0.0]], label=[1.0, -1.0])
        xgb.train(
            {"objective": "survival:cox", "tree_method": "gpu_hist"},
            d,
            num_boost_round=1,
        )
        return "gpu_hist", "GPU gpu_hist (paper §3.4)"
    except Exception:  # noqa: BLE001
        return "hist", "CPU hist fallback (XGBoost build lacks CUDA; paper: GPU)"


def optuna_xgb(
    X_tr: pd.DataFrame,
    y_tr: np.ndarray,
    X_va: pd.DataFrame,
    y_va: np.ndarray,
    n_trials: int,
    seed: int,
    hpo_cfg: dict | None = None,
) -> tuple[dict, object]:
    """
    Optuna TPE on validation cox-nloglik (paper §3.4).

    Paper-stated ceilings (§3.7 / Appendix B): max_depth ≤ 10, n_trees ≤ 2594.
    Early stopping on validation operationalizes HP selection within those ceilings.
    """
    import optuna
    import xgboost as xgb

    hpo_cfg = hpo_cfg or {}
    depth_lo, depth_hi = hpo_cfg.get("max_depth", [1, 10])
    n_max = int(hpo_cfg.get("num_boost_round_max", 2594))
    es_rounds = int(hpo_cfg.get("early_stopping_rounds", 75))
    eta_lo, eta_hi = hpo_cfg.get("eta", [0.005, 0.2])
    sub_lo, sub_hi = hpo_cfg.get("subsample", [0.5, 1.0])
    col_lo, col_hi = hpo_cfg.get("colsample_bytree", [0.5, 1.0])
    mcw_lo, mcw_hi = hpo_cfg.get("min_child_weight", [1.0, 20.0])
    lam_lo, lam_hi = hpo_cfg.get("lambda", [1e-3, 10.0])
    alp_lo, alp_hi = hpo_cfg.get("alpha", [1e-3, 10.0])
    tree_method, device_note = _xgb_tree_method(bool(hpo_cfg.get("prefer_gpu", True)))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_va, label=y_va)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "survival:cox",
            "eval_metric": "cox-nloglik",
            "tree_method": tree_method,
            "max_depth": trial.suggest_int("max_depth", int(depth_lo), int(depth_hi)),
            "eta": trial.suggest_float("eta", float(eta_lo), float(eta_hi), log=True),
            "subsample": trial.suggest_float("subsample", float(sub_lo), float(sub_hi)),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", float(col_lo), float(col_hi)
            ),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", float(mcw_lo), float(mcw_hi)
            ),
            "lambda": trial.suggest_float("lambda", float(lam_lo), float(lam_hi), log=True),
            "alpha": trial.suggest_float("alpha", float(alp_lo), float(alp_hi), log=True),
            "seed": seed,
        }
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=n_max,
            evals=[(dval, "val")],
            early_stopping_rounds=es_rounds,
            verbose_eval=False,
        )
        trial.set_user_attr("best_iteration", int(bst.best_iteration))
        trial.set_user_attr("best_ntree_limit", int(bst.best_iteration) + 1)
        return float(bst.best_score)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = dict(study.best_params)
    best_trial = study.best_trial
    n_trees = int(best_trial.user_attrs.get("best_ntree_limit", n_max))
    best["num_boost_round"] = n_trees

    params = {
        "objective": "survival:cox",
        "eval_metric": "cox-nloglik",
        "tree_method": tree_method,
        "max_depth": best["max_depth"],
        "eta": best["eta"],
        "subsample": best["subsample"],
        "colsample_bytree": best["colsample_bytree"],
        "min_child_weight": best["min_child_weight"],
        "lambda": best["lambda"],
        "alpha": best["alpha"],
        "seed": seed,
    }
    # Refit on train only (paper §3.4), with n_trees selected on validation
    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=n_trees,
        verbose_eval=False,
    )
    return {
        "best_params": best,
        "best_value": float(study.best_value),
        "n_trials": n_trials,
        "tree_method": tree_method,
        "device_note": device_note,
        "paper_bounds": {
            "max_depth_max": int(depth_hi),
            "num_boost_round_max": n_max,
            "source": "paper §3.7 / Appendix B (up to depth 10, up to 2594 trees)",
        },
        "early_stopping_rounds": es_rounds,
        "selected_num_boost_round": n_trees,
    }, bst


def main() -> int:
    log("─" * 60)
    log("D03_DOMAIN_02 — TRAIN COX / XGB-Optuna / RATINGS")
    log("─" * 60)

    path = cfg.DIRS["processed_d2"] / "loans_split.parquet"
    if not path.exists():
        log("ERROR: run D02 first.")
        return 1
    df = pd.read_parquet(path)
    feats = design_cols(df)
    log(f"  Features ({len(feats)}): {feats[:8]}{'…' if len(feats) > 8 else ''}")

    train = df.loc[df["split_role"] == "train"].copy()
    val = df.loc[df["split_role"] == "val"].copy()
    test = df.loc[df["split_role"] == "test"].copy()
    log(f"  train={len(train):,}  val={len(val):,}  test={len(test):,}")

    def frame(split_df: pd.DataFrame, use_split_censor: bool) -> pd.DataFrame:
        X = split_df[feats].apply(pd.to_numeric, errors="coerce")
        if use_split_censor:
            y = pd.DataFrame(
                {
                    "duration_days": split_df["duration_days_split"],
                    "event": split_df["event_split"],
                },
                index=split_df.index,
            )
        else:
            y = pd.DataFrame(
                {
                    "duration_days": split_df["duration_days"],
                    "event": split_df["event"],
                },
                index=split_df.index,
            )
        return pd.concat([X, y], axis=1).dropna()

    tr = frame(train, use_split_censor=True)
    va = frame(val, use_split_censor=True)

    te_X = test[feats].apply(pd.to_numeric, errors="coerce")
    meta_cols = [
        c
        for c in (
            "duration_days",
            "event",
            "Rating",
            "Status",
            "LastPaymentOn",
            "early_repayment",
            "ContractEndDate",
            "DefaultDate",
            "obs_end",
        )
        if c in test.columns
    ]
    if "Interest" not in feats and "Interest" in test.columns:
        meta_cols.append("Interest")
    te = pd.concat([te_X, test[meta_cols]], axis=1).dropna(subset=feats)
    term_days = float(round(cfg.DOMAIN_02["loan_duration_months"] * 365.25 / 12))
    as_of = pd.Timestamp(cfg.DOMAIN_02["paper_retrieve_date"])
    if "obs_end" in te.columns:
        reported = pd.to_datetime(te["obs_end"], errors="coerce").max()
        if pd.notna(reported):
            as_of = pd.Timestamp(reported)
    log(f"  complete train={len(tr):,}  val={len(va):,}  test_pred={len(te):,}")
    log(f"  term_days for KM={term_days:.1f}  completed_as_of={as_of.date()}")

    # --- Linear Cox ---
    cph = CoxPHFitter(penalizer=0.01)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cph.fit(tr, duration_col="duration_days", event_col="event", show_progress=False)
    for w in caught:
        if "Convergence" in str(w.message) or "Convergence" in w.category.__name__:
            log(f"  Warning: {w.message}")

    train_hr = cph.predict_partial_hazard(tr[feats]).values.ravel()
    test_hr = cph.predict_partial_hazard(te[feats]).values.ravel()
    c_train = float(concordance_index(tr["duration_days"], -train_hr, tr["event"]))
    c_test = float(concordance_index(te["duration_days"], -test_hr, te["event"]))
    log(f"  Cox C-index train={c_train:.4f}  test={c_test:.4f}")

    n_strata = int(cfg.DOMAIN_02["n_rating_strata"])
    labels = list(cfg.DOMAIN_02.get("rating_labels") or ["AA", "A", "B", "C", "D", "E", "F"])[
        :n_strata
    ]
    edges = np.unique(np.quantile(train_hr, np.linspace(0, 1, n_strata + 1)))
    if len(edges) < 3:
        edges = np.quantile(train_hr, np.linspace(0, 1, n_strata + 1))
    lab = labels[: max(len(edges) - 1, 1)]

    te = te.copy()
    te["rating_cox_linear"] = assign_ratings(test_hr, edges, lab)
    te["event"] = te["event"].astype(int)
    completed = mark_completed(te, as_of=as_of)
    te["is_completed"] = completed.astype(int)
    log(
        f"  Dömötör completed on test: {int(completed.sum()):,}/{len(te):,} "
        f"({100.0 * float(completed.mean()):.1f}%)  strata={lab}"
    )
    rates_cox = rating_performance_table(te, "rating_cox_linear", term_days, completed)
    if "Rating" in te.columns:
        te_b = te.dropna(subset=["Rating"])
        rates_bondora = rating_performance_table(
            te_b, "Rating", term_days, completed.loc[te_b.index]
        )
    else:
        rates_bondora = {}
    aa_b = rates_bondora.get("AA") or {}
    log(
        f"  Bondora AA defaults: completed={aa_b.get('default_rate_completed')} "
        f"(n_c={aa_b.get('n_completed')}/{aa_b.get('n')})  "
        f"empirical={aa_b.get('default_rate_empirical')}  "
        f"KM@term={aa_b.get('default_rate_km_at_term')}"
    )

    km_summary = {}
    for r in lab[:2] + lab[-2:]:
        g = te.loc[te["rating_cox_linear"] == r]
        if len(g) < 5:
            continue
        km = KaplanMeierFitter()
        km.fit(g["duration_days"], g["event"], label=r)
        km_summary[r] = {
            "n": int(len(g)),
            "km_survival_at_term": float(km.predict(min(term_days, float(g["duration_days"].max())))),
            "km_default_at_term": float(
                1.0 - km.predict(min(term_days, float(g["duration_days"].max())))
            ),
        }

    # --- XGB-Cox + Optuna on validation ---
    xgb_metrics: dict = {"status": "skipped"}
    n_trials = int(cfg.DOMAIN_02.get("optuna_trials", 80))
    hpo_cfg = dict(cfg.DOMAIN_02.get("xgb_hpo") or {})
    try:
        import xgboost as xgb

        y_tr = make_cox_label(tr["duration_days"].to_numpy(), tr["event"].to_numpy())
        y_va = make_cox_label(va["duration_days"].to_numpy(), va["event"].to_numpy())
        log(
            f"  Optuna XGB-Cox: {n_trials} trials; "
            f"depth≤{hpo_cfg.get('max_depth', [1, 10])[1]}, "
            f"trees≤{hpo_cfg.get('num_boost_round_max', 2594)} "
            f"(paper §3.7), early_stopping={hpo_cfg.get('early_stopping_rounds', 75)}"
        )
        hpo, bst = optuna_xgb(
            tr[feats],
            y_tr,
            va[feats],
            y_va,
            n_trials,
            cfg.RANDOM_SEED,
            hpo_cfg=hpo_cfg,
        )
        log(f"  XGB device: {hpo.get('device_note')}")
        dtest = xgb.DMatrix(te[feats])
        dtrain = xgb.DMatrix(tr[feats], label=y_tr)
        test_xgb = bst.predict(dtest)
        train_xgb = bst.predict(dtrain)
        edges_x = np.unique(np.quantile(train_xgb, np.linspace(0, 1, len(lab) + 1)))
        te["rating_cox_xgb"] = assign_ratings(test_xgb, edges_x, lab[: max(len(edges_x) - 1, 1)])
        rates_xgb = rating_performance_table(te, "rating_cox_xgb", term_days, completed)
        c_xgb = float(concordance_index(te["duration_days"], -test_xgb, te["event"]))
        aa_x = rates_xgb.get("AA") or {}
        log(
            f"  XGB AA defaults: completed={aa_x.get('default_rate_completed')} "
            f"(n_c={aa_x.get('n_completed')}/{aa_x.get('n')})  "
            f"empirical={aa_x.get('default_rate_empirical')}  "
            f"KM@term={aa_x.get('default_rate_km_at_term')}"
        )
        bp = hpo.get("best_params") or {}
        xgb_metrics = {
            "status": "ok",
            "cindex_test": c_xgb,
            "default_rates_by_rating": rates_xgb,
            "optuna": hpo,
            "note": (
                "Optuna TPE on validation cox-nloglik; "
                f"paper ceilings depth≤10 / trees≤2594; "
                f"selected depth={bp.get('max_depth')} trees={bp.get('num_boost_round')}; "
                f"{hpo.get('device_note')}"
            ),
        }
        log(
            f"  XGB-Cox C-index test={c_xgb:.4f}  "
            f"best_val_nloglik={hpo['best_value']:.4f}  "
            f"depth={bp.get('max_depth')} trees={bp.get('num_boost_round')}"
        )
        joblib.dump(bst, cfg.DIRS["models"] / "domain2" / "xgb_cox.joblib")
    except Exception as exc:  # noqa: BLE001
        xgb_metrics = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        log(f"  XGB-Cox failed: {xgb_metrics['error']}")

    out_dir = cfg.DIRS["models"] / "domain2"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": cph, "features": feats, "rating_edges": edges.tolist(), "labels": lab},
        out_dir / "cox_linear.joblib",
    )

    metrics = {
        "stage": "D03_DOMAIN_02",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source_url": cfg.DOMAIN_02["data_source_url"],
        "features": feats,
        "n_features": len(feats),
        "cox_linear": {
            "cindex_train": c_train,
            "cindex_test": c_test,
            "default_rates_by_rating": rates_cox,
            "rating_edges": edges.tolist(),
            "labels": lab,
        },
        "bondora_rating_on_test": rates_bondora,
        "xgb_cox": xgb_metrics,
        "km_summary": km_summary,
        "completed_definition": {
            "rule": "Bondora-closed (Repaid/Closed/Finished) OR LastPaymentOn missing/≤ as_of−365d",
            "source": "paper Table 1 / footnote 14 (Dömötör et al. 2023)",
            "as_of": str(as_of.date()),
            "inactive_days": int(cfg.DOMAIN_02.get("completed_inactive_days", 365)),
            "n_test": int(len(te)),
            "n_completed": int(completed.sum()),
        },
        "n_rating_strata": n_strata,
        "irr": {
            "status": "skipped",
            "reason": "Repayments dataset unavailable (CODE_ACCESS.md)",
        },
        "protocol_deviations": [
            "LoanData from Kaggle marcobeyer/bondora-p2p-loans aligned to 2024-01-03",
            "§3.2 preprocess implemented best-effort (no author code) — Table 4 auction-time design",
            "XGB HPO: Optuna TPE on val; paper ceilings depth≤10 / trees≤2594 (§3.7); early stopping for n_trees",
            "XGB device: CPU hist fallback if no CUDA (paper: GPU §3.4)",
            "eta/subsample/reg ranges not published — standard XGB search",
            "IRR skipped — no repayments file",
            "Ratings: 7 equal-mass HR bins on train (AA–F; paper Table 1 / Bondora count)",
            "Table 1 defaults: Dömötör completed (closed OR ≥1y without payment)",
        ],
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    te.to_parquet(out_dir / "test_scored.parquet", index=False)

    log(f"  Wrote {out_dir.relative_to(cfg.ROOT)}/metrics.json")
    log("D03_DOMAIN_02 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
