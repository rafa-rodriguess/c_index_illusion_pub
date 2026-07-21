"""
Breslow baseline for XGBoost Cox (absolute S(t|x) from frozen Booster).

XGB ``survival:cox`` exposes a risk score η(x) (higher = higher hazard).
Given training (time, event, η), Breslow estimates H₀(t); then

    S(t | x) = exp( - H₀(t) · exp(η(x) - η̄_train) )

Centering η on the training mean stabilizes the baseline (same as common
Cox partial-hazard practice).
"""

from __future__ import annotations

from typing import Any

import numpy as np


def breslow_cumulative_hazard(
    time: np.ndarray,
    event: np.ndarray,
    log_risk: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (event_times, H0_at_event_times) via Breslow.

    Only unique event times (event==1) appear in the output grid.
    """
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    eta = np.asarray(log_risk, dtype=float)
    if len(time) != len(event) or len(time) != len(eta):
        raise ValueError("time/event/log_risk length mismatch")
    if int(event.sum()) < 1:
        raise ValueError("Breslow needs at least one event")

    # Center for numerical stability
    eta_c = eta - float(np.mean(eta))
    risk = np.exp(np.clip(eta_c, -50.0, 50.0))

    order = np.argsort(time, kind="mergesort")
    t_sorted = time[order]
    e_sorted = event[order]
    r_sorted = risk[order]

    # Reverse cumulative sum of risk for risk-set totals
    # risk_set_sum[i] = sum of r for subjects with time >= t_sorted[i]
    rev_cumsum = np.cumsum(r_sorted[::-1])[::-1]

    event_times: list[float] = []
    H0: list[float] = []
    h = 0.0
    n = len(t_sorted)
    i = 0
    while i < n:
        if e_sorted[i] != 1:
            i += 1
            continue
        ti = t_sorted[i]
        # ties at ti
        j = i
        d_i = 0
        while j < n and t_sorted[j] == ti:
            if e_sorted[j] == 1:
                d_i += 1
            j += 1
        denom = float(rev_cumsum[i])
        if denom <= 0:
            denom = 1e-300
        h += d_i / denom
        event_times.append(float(ti))
        H0.append(float(h))
        i = j

    return np.asarray(event_times, dtype=float), np.asarray(H0, dtype=float)


def survival_from_breslow(
    times_grid: np.ndarray,
    event_times: np.ndarray,
    H0: np.ndarray,
    log_risk: np.ndarray,
    eta_train_mean: float,
) -> np.ndarray:
    """
    S(t|x) on ``times_grid`` for each subject.

    Returns array shape (n_times, n_subjects).
    """
    times_grid = np.asarray(times_grid, dtype=float)
    eta = np.asarray(log_risk, dtype=float)
    eta_c = eta - float(eta_train_mean)
    # H0(t) step function via searchsorted
    # right-continuous: H0 at t = last H0 with event_time <= t
    idx = np.searchsorted(event_times, times_grid, side="right") - 1
    H0_t = np.zeros(len(times_grid), dtype=float)
    valid = idx >= 0
    H0_t[valid] = H0[idx[valid]]
    # S = exp(-H0 * exp(eta_c))
    scale = np.exp(np.clip(eta_c, -50.0, 50.0))  # (n,)
    # (n_times, n)
    cumhaz = H0_t[:, None] * scale[None, :]
    return np.exp(-np.clip(cumhaz, 0.0, 50.0))


def fit_breslow_bundle(
    time_train: np.ndarray,
    event_train: np.ndarray,
    log_risk_train: np.ndarray,
) -> dict[str, Any]:
    eta_mean = float(np.mean(log_risk_train))
    et, H0 = breslow_cumulative_hazard(time_train, event_train, log_risk_train)
    return {
        "method": "breslow",
        "eta_train_mean": eta_mean,
        "event_times": et,
        "H0": H0,
        "n_train": int(len(time_train)),
        "n_events_train": int(np.asarray(event_train).sum()),
    }


def predict_survival_breslow(
    bundle: dict[str, Any],
    log_risk: np.ndarray,
    times_grid: np.ndarray,
    time_obs: np.ndarray | None = None,
) -> dict[str, Any]:
    et = np.asarray(bundle["event_times"], dtype=float)
    H0 = np.asarray(bundle["H0"], dtype=float)
    eta_mean = float(bundle["eta_train_mean"])
    surv_grid = survival_from_breslow(times_grid, et, H0, log_risk, eta_mean)
    surv_obs = None
    if time_obs is not None:
        time_obs = np.asarray(time_obs, dtype=float)
        eta = np.asarray(log_risk, dtype=float)
        eta_c = eta - eta_mean
        scale = np.exp(np.clip(eta_c, -50.0, 50.0))
        idx = np.searchsorted(et, time_obs, side="right") - 1
        H0_i = np.zeros(len(time_obs), dtype=float)
        valid = idx >= 0
        H0_i[valid] = H0[idx[valid]]
        surv_obs = np.exp(-np.clip(H0_i * scale, 0.0, 50.0))
    return {
        "surv_grid": surv_grid,
        "surv_at_observed": surv_obs,
        "times_grid": np.asarray(times_grid, dtype=float),
        "curve_backend": "xgb_cox_breslow",
    }
