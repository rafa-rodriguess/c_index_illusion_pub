"""
Load frozen D estimators and produce risk scores for Block E.

Risk convention for sksurv: **higher = higher event risk**.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.config import cfg


def _root() -> Path:
    return cfg.ROOT


def load_eval_frame(artifact: dict[str, Any]) -> pd.DataFrame:
    path = _root() / artifact["eval_data_path"]
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in {".csv", ".gz"} or path.name.endswith(".csv.gz"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported eval data format: {path}")


def load_estimator_blob(artifact: dict[str, Any]) -> Any:
    path = _root() / artifact["path"]
    return joblib.load(path)


def _feature_matrix(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing features in eval frame: {missing[:8]}…")
    X = df[feature_cols].copy()
    return X


def predict_risk_scores(
    artifact: dict[str, Any],
    df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Return time, event, risk (higher=worse), feature_cols, n.

    DOMAIN_01: in-sample complete cases on H6a cohort
    (``drives_cox_cohort.parquet`` = healthy calendar>7y ∪ all failed).
    DOMAIN_02: test_scored.parquet (temporal hold-out).
    """
    if artifact.get("kind") != "estimator" or not artifact.get("predict_ready"):
        raise ValueError(f"Artifact not predict-ready: {artifact.get('model_id')}")

    df = df if df is not None else load_eval_frame(artifact)
    time_col = artifact["time_col"]
    event_col = artifact["event_col"]
    blob = load_estimator_blob(artifact)

    feature_cols = artifact.get("feature_cols")
    model = blob
    if isinstance(blob, dict):
        model = blob.get("model", blob)
        feature_cols = feature_cols or blob.get("smart_cols") or blob.get("features")
    if not feature_cols:
        raise ValueError(f"No feature_cols for {artifact['model_id']}")

    # Drop incomplete / non-positive duration
    work = df.copy()
    work = work.dropna(subset=[time_col, event_col] + list(feature_cols))
    work = work.loc[work[time_col] > 0].copy()

    X = _feature_matrix(work, list(feature_cols))
    # fill residual NaNs in features with training-safe 0 (one-hots / rare)
    X = X.fillna(0.0)

    backend = artifact.get("backend") or ""
    mid = artifact["model_id"]

    if "CoxPH" in type(model).__name__ or "lifelines" in backend or mid in {
        "cox_full",
        "cox_ablated",
        "cox_classical",
    }:
        # lifelines CoxPHFitter
        risk = np.asarray(model.predict_partial_hazard(X), dtype=float).ravel()
    elif "RandomSurvivalForest" in type(model).__name__ or mid.startswith("rsf_"):
        risk = np.asarray(model.predict(X.to_numpy(dtype=float)), dtype=float).ravel()
    elif "Booster" in type(model).__name__ or "xgb" in backend.lower() or mid == "cox_xgboost":
        import xgboost as xgb

        dmat = xgb.DMatrix(X.to_numpy(dtype=np.float32), feature_names=list(feature_cols))
        # XGB Cox: higher margin ≈ higher risk (same as training concordance with -score? )
        # Domain2 training used: concordance_index(time, -test_xgb, event) ⇒ xgb score is risk-like
        # and they negate for lifelines API. sksurv wants higher=risk ⇒ use raw xgb output.
        risk = np.asarray(model.predict(dmat), dtype=float).ravel()
    else:
        raise TypeError(f"Unsupported estimator type {type(model)} for {mid}")

    time = work[time_col].to_numpy(dtype=float)
    event = work[event_col].to_numpy(dtype=int)
    return {
        "time": time,
        "event": event,
        "risk": risk,
        "feature_cols": list(feature_cols),
        "n": int(len(work)),
        "n_events": int(event.sum()),
        "model_id": mid,
        "domain_id": artifact["domain_id"],
    }


def _unpack_cox(artifact: dict[str, Any], df: pd.DataFrame | None = None) -> dict[str, Any]:
    """Shared loader for risk + lifelines CoxPHFitter (when available)."""
    if artifact.get("kind") != "estimator" or not artifact.get("predict_ready"):
        raise ValueError(f"Artifact not predict-ready: {artifact.get('model_id')}")

    df = df if df is not None else load_eval_frame(artifact)
    time_col = artifact["time_col"]
    event_col = artifact["event_col"]
    blob = load_estimator_blob(artifact)

    feature_cols = artifact.get("feature_cols")
    model = blob
    if isinstance(blob, dict):
        model = blob.get("model", blob)
        feature_cols = feature_cols or blob.get("smart_cols") or blob.get("features")
    if not feature_cols:
        raise ValueError(f"No feature_cols for {artifact['model_id']}")

    work = df.copy()
    work = work.dropna(subset=[time_col, event_col] + list(feature_cols))
    work = work.loc[work[time_col] > 0].copy()
    X = _feature_matrix(work, list(feature_cols)).fillna(0.0)
    time = work[time_col].to_numpy(dtype=float)
    event = work[event_col].to_numpy(dtype=int)
    return {
        "model": model,
        "X": X,
        "time": time,
        "event": event,
        "feature_cols": list(feature_cols),
        "model_id": artifact["model_id"],
        "domain_id": artifact["domain_id"],
        "backend": artifact.get("backend") or "",
    }


def _is_lifelines_cox(model: Any, backend: str, model_id: str) -> bool:
    return (
        "CoxPH" in type(model).__name__
        or "lifelines" in (backend or "")
        or model_id in {"cox_full", "cox_ablated", "cox_classical"}
    )


def _is_sksurv_rsf(model: Any, model_id: str) -> bool:
    return "RandomSurvivalForest" in type(model).__name__ or str(model_id).startswith("rsf_")


def _is_xgb_cox(model: Any, backend: str, model_id: str) -> bool:
    name = type(model).__name__
    return (
        name == "Booster"
        or "xgb" in (backend or "").lower()
        or model_id == "cox_xgboost"
    )


def predict_survival_curves(
    artifact: dict[str, Any],
    df: pd.DataFrame | None = None,
    times_grid: np.ndarray | None = None,
    n_grid: int = 40,
) -> dict[str, Any]:
    """
    Absolute survival curves S(t|x) for estimators that expose a baseline / SF.

    Lifelines Cox: S(t|x) = S0(t) ** partial_hazard(x).
    sksurv RSF: ``predict_survival_function``.
    XGB Cox: Breslow H0 attached by D03b (``blob['breslow']``).
    """
    pack = _unpack_cox(artifact, df=df)
    model, X, time, event = pack["model"], pack["X"], pack["time"], pack["event"]
    mid, backend = pack["model_id"], pack["backend"]

    if times_grid is None:
        lo = 0.0
        hi = float(np.quantile(time, 0.95))
        if not np.isfinite(hi) or hi <= 0:
            hi = float(np.max(time))
        times_grid = np.linspace(lo, hi, n_grid)
        times_grid[0] = 0.0
    times_grid = np.asarray(times_grid, dtype=float)

    if _is_lifelines_cox(model, backend, mid):
        if not hasattr(model, "baseline_survival_"):
            raise TypeError(f"{mid}: CoxPHFitter missing baseline_survival_")
        ph = np.asarray(model.predict_partial_hazard(X), dtype=float).ravel()
        bs = model.baseline_survival_
        t0 = bs.index.to_numpy(dtype=float)
        s0 = np.clip(bs.to_numpy(dtype=float).ravel(), 1e-300, 1.0)
        s0_obs = np.clip(np.interp(time, t0, s0, left=1.0, right=float(s0[-1])), 1e-300, 1.0)
        surv_obs = np.exp(ph * np.log(s0_obs))
        s0_grid = np.clip(np.interp(times_grid, t0, s0, left=1.0, right=float(s0[-1])), 1e-300, 1.0)
        surv_grid = np.exp(np.log(s0_grid)[:, None] * ph[None, :])
        risk = ph
        curve_backend = "lifelines_baseline_power"
    elif _is_sksurv_rsf(model, mid):
        Xn = X.to_numpy(dtype=float)
        risk = np.asarray(model.predict(Xn), dtype=float).ravel()
        fns = model.predict_survival_function(Xn)
        surv_grid = np.vstack([fn(times_grid) for fn in fns]).T
        surv_obs = np.array([float(fn(t)) for fn, t in zip(fns, time, strict=True)], dtype=float)
        curve_backend = "sksurv_rsf_survival_function"
    elif _is_xgb_cox(model, backend, mid):
        import xgboost as xgb

        from src.metrics.xgb_breslow import predict_survival_breslow

        blob = load_estimator_blob(artifact)
        breslow = None
        if isinstance(blob, dict):
            breslow = blob.get("breslow")
        if not breslow:
            raise TypeError(
                f"{mid}: XGB Booster has no Breslow bundle — run "
                "D03b_DOMAIN_02_export_xgb_breslow.py"
            )
        # Ensure arrays
        bundle = {
            "event_times": np.asarray(breslow["event_times"], dtype=float),
            "H0": np.asarray(breslow["H0"], dtype=float),
            "eta_train_mean": float(breslow["eta_train_mean"]),
        }
        feats = pack["feature_cols"]
        dmat = xgb.DMatrix(X.to_numpy(dtype=np.float32), feature_names=list(feats))
        risk = np.asarray(model.predict(dmat), dtype=float).ravel()
        pred = predict_survival_breslow(bundle, risk, times_grid, time_obs=time)
        surv_grid = pred["surv_grid"]
        surv_obs = pred["surv_at_observed"]
        curve_backend = "xgb_cox_breslow"
    else:
        raise TypeError(
            f"No absolute survival curve for {mid} ({type(model).__name__}); "
            "need Cox baseline, RSF survival function, or XGB Breslow bundle."
        )

    return {
        "time": time,
        "event": event,
        "risk": risk,
        "surv_at_observed": surv_obs,
        "times_grid": times_grid,
        "surv_grid": surv_grid,
        "feature_cols": pack["feature_cols"],
        "n": int(len(time)),
        "n_events": int(event.sum()),
        "model_id": mid,
        "domain_id": pack["domain_id"],
        "curve_backend": curve_backend,
    }


def load_train_for_ipcw(artifact: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    """Train (time, event) for Uno IPCW. D1: same pop; D2: split_role==train; D3: panel train fold."""
    domain = artifact["domain_id"]
    if domain == "DOMAIN_01":
        # In-sample GOF — use eval frame itself as "train" for G_hat
        pred = predict_risk_scores(artifact)
        return pred["time"], pred["event"]
    if domain == "DOMAIN_02":
        path = _root() / "data/processed/domain2/loans_split.parquet"
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        tr = df.loc[df["split_role"] == "train"].copy()
        # paper uses duration_days_split / event_split on train
        tcol = "duration_days_split" if "duration_days_split" in tr.columns else "duration_days"
        ecol = "event_split" if "event_split" in tr.columns else "event"
        tr = tr.dropna(subset=[tcol, ecol])
        tr = tr.loc[tr[tcol] > 0]
        return tr[tcol].to_numpy(dtype=float), tr[ecol].to_numpy(dtype=int)
    if domain == "DOMAIN_03":
        # Use full Politics θ=24 panel minus eval rows as IPCW train proxy
        panel = _root() / "data/processed/domain3/p_theta24.parquet"
        eval_path = _root() / artifact.get("eval_data_path", "")
        if not panel.exists():
            return None
        df = pd.read_parquet(panel)
        tcol = artifact.get("time_col") or "duration_months"
        ecol = artifact.get("event_col") or "event"
        if eval_path.exists() and "UserId" in df.columns:
            ev = pd.read_parquet(eval_path)
            if "UserId" in ev.columns:
                df = df.loc[~df["UserId"].isin(set(ev["UserId"]))].copy()
        df = df.dropna(subset=[tcol, ecol])
        df = df.loc[df[tcol] > 0]
        return df[tcol].to_numpy(dtype=float), df[ecol].to_numpy(dtype=int)
    return None
