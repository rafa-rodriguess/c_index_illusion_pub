"""
Time-dependent AUC (cumulative/dynamic) for H5.

Uses sksurv ``cumulative_dynamic_auc`` (Uno-style IPCW cum/dyn AUC).
Primary estimate for RSF: risk at horizon t via ``1 - S(t|x)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
from sksurv.util import Surv

from src.config import cfg


def _surv_y(time: np.ndarray, event: np.ndarray):
    return Surv.from_arrays(event=event.astype(bool), time=time.astype(float))


def harrell_c(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    return float(
        concordance_index_censored(
            event.astype(bool), time.astype(float), risk.astype(float)
        )[0]
    )


def auc_at_horizons(
    *,
    time_train: np.ndarray,
    event_train: np.ndarray,
    time_test: np.ndarray,
    event_test: np.ndarray,
    estimate: np.ndarray,
    horizons: list[float],
) -> dict[str, Any]:
    """
    Cumulative/dynamic AUC at each horizon.

    ``estimate`` shape (n_test,) for a single risk score, or
    (n_test, n_horizons) for time-dependent risk (one column per horizon).
    """
    y_train = _surv_y(time_train, event_train)
    y_test = _surv_y(time_test, event_test)
    times = np.asarray(horizons, dtype=float)
    # sksurv requires times strictly within follow-up of test set
    t_max = float(np.max(time_test))
    usable = times[times < t_max]
    if usable.size == 0:
        return {
            "metric": "cumulative_dynamic_auc",
            "status": "error",
            "error": f"all horizons >= max test time ({t_max})",
            "values": {},
            "mean_auc": None,
        }
    try:
        aucs, mean_auc = cumulative_dynamic_auc(
            y_train, y_test, estimate, usable
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "metric": "cumulative_dynamic_auc",
            "status": "error",
            "error": str(exc),
            "values": {},
            "mean_auc": None,
        }

    values: dict[str, float | None] = {str(int(h) if float(h).is_integer() else h): None for h in horizons}
    for t, a in zip(usable, aucs, strict=True):
        key = str(int(t) if float(t).is_integer() else t)
        values[key] = float(a)
    return {
        "metric": "cumulative_dynamic_auc",
        "status": "live",
        "backend": "sksurv.metrics.cumulative_dynamic_auc",
        "horizons": [float(h) for h in horizons],
        "values": values,
        "mean_auc": float(mean_auc),
        "t_max_test": t_max,
    }


def risk_at_horizons_from_surv(
    surv_grid: np.ndarray,
    times_grid: np.ndarray,
    horizons: list[float],
) -> np.ndarray:
    """
    Build (n, n_horizons) risk = 1 - S(t) by interpolating surv_grid (n_times, n).
    """
    s = np.asarray(surv_grid, dtype=float)
    tg = np.asarray(times_grid, dtype=float)
    n = s.shape[1]
    out = np.empty((n, len(horizons)), dtype=float)
    for j, h in enumerate(horizons):
        # Linear interp along the time axis for every subject
        sh = np.apply_along_axis(
            lambda col, hh=h: np.interp(hh, tg, col, left=1.0, right=float(col[-1])),
            0,
            s,
        )
        out[:, j] = 1.0 - np.clip(sh, 0.0, 1.0)
    return out


def h5_auc_strip(
    *,
    time_train: np.ndarray,
    event_train: np.ndarray,
    time_test: np.ndarray,
    event_test: np.ndarray,
    risk: np.ndarray,
    surv_grid: np.ndarray | None = None,
    times_grid: np.ndarray | None = None,
    horizons_months: list[int] | None = None,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """
    H5 core: AUC(t) at configured months + bootstrap SE on Δ = AUC(12)−AUC(36).

    Prefers time-dependent risk from survival curves when available; else static risk.
    """
    h5 = cfg.PROTOCOL["hypotheses"]["H5"]
    horizons = [float(m) for m in (horizons_months or h5["horizons_months"])]
    band = list(h5["global_cindex_band"])

    if surv_grid is not None and times_grid is not None:
        est = risk_at_horizons_from_surv(surv_grid, times_grid, horizons)
        est_mode = "time_dependent_1_minus_S"
    else:
        est = np.asarray(risk, dtype=float).ravel()
        est_mode = "static_risk"

    point = auc_at_horizons(
        time_train=time_train,
        event_train=event_train,
        time_test=time_test,
        event_test=event_test,
        estimate=est,
        horizons=horizons,
    )

    global_c = harrell_c(time_test, event_test, risk)
    in_band = bool(band[0] <= global_c <= band[1])

    # Pointwise Δ and monotonicity
    vals = point.get("values") or {}
    keys = [str(int(h)) for h in horizons]
    series = [vals.get(k) for k in keys]
    live_pairs = [(k, v) for k, v in zip(keys, series) if v is not None]
    mono_decay = False
    delta_12_36 = None
    if len(live_pairs) >= 2:
        nums = [v for _, v in live_pairs]
        mono_decay = all(nums[i] >= nums[i + 1] for i in range(len(nums) - 1))
    if vals.get("12") is not None and vals.get("36") is not None:
        delta_12_36 = float(vals["12"]) - float(vals["36"])

    # Bootstrap Δ and per-horizon AUC
    rng = np.random.default_rng(seed)
    n = len(time_test)
    boot_delta: list[float] = []
    boot_auc: dict[str, list[float]] = {k: [] for k in keys}
    B = int(n_bootstrap)
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        tt = time_test[idx]
        et = event_test[idx]
        if est.ndim == 1:
            ei = est[idx]
        else:
            ei = est[idx, :]
        # Need events and enough follow-up
        if int(et.sum()) < 5:
            continue
        try:
            res = auc_at_horizons(
                time_train=time_train,
                event_train=event_train,
                time_test=tt,
                event_test=et,
                estimate=ei,
                horizons=horizons,
            )
        except Exception:  # noqa: BLE001
            continue
        if res.get("status") != "live":
            continue
        bv = res["values"]
        for k in keys:
            if bv.get(k) is not None:
                boot_auc[k].append(float(bv[k]))
        if bv.get("12") is not None and bv.get("36") is not None:
            boot_delta.append(float(bv["12"]) - float(bv["36"]))

    def _se(xs: list[float]) -> float | None:
        if len(xs) < 2:
            return None
        return float(np.std(xs, ddof=1))

    se_delta = _se(boot_delta)
    se_12 = _se(boot_auc["12"])
    se_36 = _se(boot_auc["36"])
    # Protocol: AUC(12)−AUC(36) > 2 combined SEs
    combined_se = None
    if se_12 is not None and se_36 is not None:
        combined_se = float(np.sqrt(se_12**2 + se_36**2))
    elif se_delta is not None:
        combined_se = se_delta

    drop_sig = False
    if delta_12_36 is not None and combined_se is not None and combined_se > 0:
        drop_sig = bool(delta_12_36 > 2.0 * combined_se)

    h5_reject = bool(mono_decay and drop_sig and in_band)

    return {
        "metric": "h5_auc_strip",
        "status": point.get("status", "error"),
        "estimate_mode": est_mode,
        "horizons_months": [int(h) for h in horizons],
        "auc": point,
        "global_harrell_c": global_c,
        "global_cindex_band": band,
        "global_c_in_band": in_band,
        "monotonic_decay": mono_decay,
        "delta_auc_12_minus_36": delta_12_36,
        "bootstrap": {
            "B": B,
            "n_valid_delta": len(boot_delta),
            "se_delta": se_delta,
            "se_auc_12": se_12,
            "se_auc_36": se_36,
            "combined_se": combined_se,
            "ci_delta_low": float(np.percentile(boot_delta, 2.5)) if boot_delta else None,
            "ci_delta_high": float(np.percentile(boot_delta, 97.5)) if boot_delta else None,
            "mean_delta": float(np.mean(boot_delta)) if boot_delta else None,
        },
        "drop_exceeds_2se": drop_sig,
        "h5_auc_would_reject": h5_reject,
        "h5_preview_reject": h5_reject,  # legacy alias; F02 decision uses Brier primary
        "error": point.get("error"),
    }


def _surv_at_horizons(
    surv_grid: np.ndarray,
    times_grid: np.ndarray,
    horizons: list[float],
) -> np.ndarray:
    """Interpolate S(t) → shape (n, n_horizons)."""
    s = np.asarray(surv_grid, dtype=float)
    tg = np.asarray(times_grid, dtype=float)
    cols = []
    for h in horizons:
        sh = np.apply_along_axis(
            lambda col, hh=h: np.interp(hh, tg, col, left=1.0, right=float(col[-1])),
            0,
            s,
        )
        cols.append(np.clip(sh, 0.0, 1.0))
    return np.column_stack(cols)


def brier_at_horizons_sksurv(
    *,
    time_train: np.ndarray,
    event_train: np.ndarray,
    time_test: np.ndarray,
    event_test: np.ndarray,
    surv_at_horizons: np.ndarray,
    horizons: list[float],
) -> dict[str, Any]:
    """Pointwise IPCW Brier via sksurv (fast path for H5 bootstrap)."""
    from sksurv.metrics import brier_score

    y_train = _surv_y(time_train, event_train)
    y_test = _surv_y(time_test, event_test)
    times = np.asarray(horizons, dtype=float)
    try:
        t_out, scores = brier_score(
            y_train, y_test, np.asarray(surv_at_horizons, dtype=float), times
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "metric": "ipcw_brier_sksurv",
            "status": "error",
            "error": str(exc),
            "values": {},
        }
    values = {
        str(int(t) if float(t).is_integer() else t): float(s)
        for t, s in zip(t_out, scores, strict=True)
    }
    return {
        "metric": "ipcw_brier_sksurv",
        "status": "live",
        "backend": "sksurv.metrics.brier_score",
        "horizons": [float(h) for h in horizons],
        "values": values,
    }


def h5_brier_strip(
    *,
    time_train: np.ndarray,
    event_train: np.ndarray,
    time_test: np.ndarray,
    event_test: np.ndarray,
    risk: np.ndarray,
    surv_grid: np.ndarray,
    times_grid: np.ndarray,
    horizons_months: list[int] | None = None,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> dict[str, Any]:
    """
    H5 primary: IPCW Brier rises across horizons while global C stays in band.
    """
    h5 = cfg.PROTOCOL["hypotheses"]["H5"]
    horizons = [float(m) for m in (horizons_months or h5["horizons_months"])]
    band = list(h5["global_cindex_band"])
    keys = [str(int(h)) for h in horizons]

    S_h = _surv_at_horizons(surv_grid, times_grid, horizons)
    point = brier_at_horizons_sksurv(
        time_train=time_train,
        event_train=event_train,
        time_test=time_test,
        event_test=event_test,
        surv_at_horizons=S_h,
        horizons=horizons,
    )
    global_c = harrell_c(time_test, event_test, risk)
    in_band = bool(band[0] <= global_c <= band[1])

    vals = point.get("values") or {}
    series = [vals.get(k) for k in keys]
    mono_increase = False
    delta_36_12 = None
    if all(v is not None for v in series) and len(series) >= 2:
        nums = [float(v) for v in series]  # type: ignore[arg-type]
        mono_increase = all(nums[i] <= nums[i + 1] for i in range(len(nums) - 1))
    if vals.get("12") is not None and vals.get("36") is not None:
        delta_36_12 = float(vals["36"]) - float(vals["12"])

    rng = np.random.default_rng(seed)
    n = len(time_test)
    boot_delta: list[float] = []
    boot_b: dict[str, list[float]] = {k: [] for k in keys}
    B = int(n_bootstrap)
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        if int(event_test[idx].sum()) < 5:
            continue
        res = brier_at_horizons_sksurv(
            time_train=time_train,
            event_train=event_train,
            time_test=time_test[idx],
            event_test=event_test[idx],
            surv_at_horizons=S_h[idx, :],
            horizons=horizons,
        )
        if res.get("status") != "live":
            continue
        bv = res["values"]
        for k in keys:
            if bv.get(k) is not None:
                boot_b[k].append(float(bv[k]))
        if bv.get("12") is not None and bv.get("36") is not None:
            boot_delta.append(float(bv["36"]) - float(bv["12"]))

    def _se(xs: list[float]) -> float | None:
        if len(xs) < 2:
            return None
        return float(np.std(xs, ddof=1))

    se_delta = _se(boot_delta)
    se_12 = _se(boot_b["12"])
    se_36 = _se(boot_b["36"])
    combined_se = None
    if se_12 is not None and se_36 is not None:
        combined_se = float(np.sqrt(se_12**2 + se_36**2))
    elif se_delta is not None:
        combined_se = se_delta

    rise_sig = False
    if delta_36_12 is not None and combined_se is not None and combined_se > 0:
        rise_sig = bool(delta_36_12 > 2.0 * combined_se)

    h5_reject = bool(mono_increase and rise_sig and in_band)

    return {
        "metric": "h5_brier_strip",
        "status": point.get("status", "error"),
        "backend": "sksurv.metrics.brier_score",
        "horizons_months": [int(h) for h in horizons],
        "brier": point,
        "global_harrell_c": global_c,
        "global_cindex_band": band,
        "global_c_in_band": in_band,
        "monotonic_increase": mono_increase,
        "delta_brier_36_minus_12": delta_36_12,
        "bootstrap": {
            "B": B,
            "n_valid_delta": len(boot_delta),
            "se_delta": se_delta,
            "se_brier_12": se_12,
            "se_brier_36": se_36,
            "combined_se": combined_se,
            "ci_delta_low": float(np.percentile(boot_delta, 2.5)) if boot_delta else None,
            "ci_delta_high": float(np.percentile(boot_delta, 97.5)) if boot_delta else None,
            "mean_delta": float(np.mean(boot_delta)) if boot_delta else None,
        },
        "rise_exceeds_2se": rise_sig,
        "h5_preview_reject": h5_reject,
        "error": point.get("error"),
    }
