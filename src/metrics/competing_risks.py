"""
Rung 5 — Competing risks (CIF bias core for H3; FG / Wolbers when ``*_cr`` exists).

Primary live path (no retrain): Bondora default vs early-repayment
  - naive CIF = 1 − KM treating early repayment as censoring
  - AJ CIF = Aalen–Johansen with causes {default=1, early=2}
  - Δ = F_naive(τ) − F_AJ(τ) at 12 months (H3)

Cause-specific / Fine–Gray / Wolbers remain stub until separate ``*_cr``
artifacts are trained (must not overwrite Fase A baselines).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lifelines import AalenJohansenFitter, KaplanMeierFitter

from src.config import cfg

DAYS_PER_MONTH = 365.25 / 12.0
CAUSE_DEFAULT = 1
CAUSE_EARLY = 2


def build_cr_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map Bondora Fase A labels to competing-risk codes.

    event==1 → default (cause 1); early_repayment==1 → competitor (cause 2);
    else → censored (0).

    Timing note: Fase A sets early-repayment ``duration_days`` to the contractual
    term (censoring under the paper's single-event lens). For CR we recover the
    actual early-close time from ``ContractEndDate - LoanDate`` when available.
    """
    need = {"duration_days", "event", "early_repayment"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"CR frame missing columns: {sorted(missing)}")

    out = df.copy()
    t = out["duration_days"].to_numpy(dtype=float).copy()
    event = out["event"].to_numpy(dtype=int)
    early = out["early_repayment"].to_numpy(dtype=int)
    cr = np.zeros(len(out), dtype=int)
    cr[event == 1] = CAUSE_DEFAULT
    cr[(early == 1) & (event == 0)] = CAUSE_EARLY

    # Recover competing-event clock for early repayments
    if {"LoanDate", "ContractEndDate"}.issubset(out.columns):
        loan = pd.to_datetime(out["LoanDate"], errors="coerce")
        end = pd.to_datetime(out["ContractEndDate"], errors="coerce")
        t_early = (end - loan).dt.days.to_numpy(dtype=float)
        m = (cr == CAUSE_EARLY) & np.isfinite(t_early) & (t_early > 0)
        t[m] = t_early[m]

    out["cr_event"] = cr
    out["cr_time"] = t
    return out.loc[np.isfinite(t) & (t > 0)].copy()


def _cif_at(
    times: np.ndarray,
    cif: np.ndarray,
    horizon: float,
) -> float:
    if len(times) == 0:
        return float("nan")
    idx = np.searchsorted(times, horizon, side="right") - 1
    if idx < 0:
        return 0.0
    return float(cif[idx])


def naive_default_cif(
    time: np.ndarray,
    cr_event: np.ndarray,
    horizon: float,
) -> dict[str, Any]:
    """1−KM with default as event; early repayment (and other) as censoring."""
    time = np.asarray(time, dtype=float)
    cr_event = np.asarray(cr_event, dtype=int)
    default = (cr_event == CAUSE_DEFAULT).astype(int)
    kmf = KaplanMeierFitter()
    kmf.fit(time, event_observed=default)
    surv = kmf.survival_function_
    t = surv.index.to_numpy(dtype=float)
    s = surv.iloc[:, 0].to_numpy(dtype=float)
    cif = 1.0 - s
    return {
        "metric": "naive_default_cif",
        "status": "live",
        "method": "1_minus_km_early_as_censor",
        "horizon_days": float(horizon),
        "value": _cif_at(t, cif, horizon),
        "n": int(len(time)),
        "n_default": int(default.sum()),
        "n_early": int((cr_event == CAUSE_EARLY).sum()),
        "n_censor": int((cr_event == 0).sum()),
    }


def aj_default_cif(
    time: np.ndarray,
    cr_event: np.ndarray,
    horizon: float,
) -> dict[str, Any]:
    """Aalen–Johansen CIF for default with early repayment as competing event."""
    time = np.asarray(time, dtype=float)
    cr_event = np.asarray(cr_event, dtype=int)
    aj = AalenJohansenFitter(calculate_variance=False)
    aj.fit(time, cr_event, event_of_interest=CAUSE_DEFAULT)
    cif_df = aj.cumulative_density_
    t = cif_df.index.to_numpy(dtype=float)
    cif = cif_df.iloc[:, 0].to_numpy(dtype=float)
    return {
        "metric": "aj_default_cif",
        "status": "live",
        "method": "aalen_johansen_lifelines",
        "horizon_days": float(horizon),
        "value": _cif_at(t, cif, horizon),
        "n": int(len(time)),
        "n_default": int((cr_event == CAUSE_DEFAULT).sum()),
        "n_early": int((cr_event == CAUSE_EARLY).sum()),
        "n_censor": int((cr_event == 0).sum()),
    }


def cif_bias(
    time: np.ndarray,
    cr_event: np.ndarray,
    *,
    horizon_months: float | None = None,
    n_bootstrap: int | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """
    H3 core: Δ = F_naive(τ) − F_AJ(τ). Expected sign > 0 under CR blindness.
    """
    h_m = float(
        horizon_months
        if horizon_months is not None
        else cfg.PROTOCOL["hypotheses"]["H3"]["horizon_months"]
    )
    horizon = h_m * DAYS_PER_MONTH
    delta_thr = float(cfg.PROTOCOL["hypotheses"]["H3"]["delta"])
    B = int(
        n_bootstrap
        if n_bootstrap is not None
        else min(int(cfg.EVAL["n_bootstrap"]), 1000)
    )

    naive = naive_default_cif(time, cr_event, horizon)
    aj = aj_default_cif(time, cr_event, horizon)
    delta = float(naive["value"] - aj["value"])

    boot_block: dict[str, Any]
    if B <= 0:
        boot_block = {"B": 0, "ci_low": None, "ci_high": None, "ci_excludes_0": None}
        ci_excludes_0 = False
        directional = bool(delta > delta_thr)
        reject = bool(abs(delta) > delta_thr)
    else:
        rng = np.random.default_rng(seed)
        n = len(time)
        boots = np.empty(B, dtype=float)
        for b in range(B):
            idx = rng.integers(0, n, size=n)
            nb = naive_default_cif(time[idx], cr_event[idx], horizon)["value"]
            ab = aj_default_cif(time[idx], cr_event[idx], horizon)["value"]
            boots[b] = nb - ab
        lo, hi = np.quantile(boots, [0.025, 0.975])
        ci_excludes_0 = bool(lo > 0 or hi < 0)
        directional = bool(delta > delta_thr and lo > 0)
        reject = bool(abs(delta) > delta_thr and ci_excludes_0)
        boot_block = {
            "B": B,
            "ci_low": float(lo),
            "ci_high": float(hi),
            "ci_excludes_0": ci_excludes_0,
            "mean": float(np.mean(boots)),
        }

    return {
        "metric": "cif_naive_minus_aj",
        "status": "live",
        "horizon_months": h_m,
        "horizon_days": horizon,
        "naive": naive,
        "aj": aj,
        "delta": delta,
        "abs_delta": abs(delta),
        "delta_threshold": delta_thr,
        "bootstrap": boot_block,
        "reject_h0_pointwise": reject,
        "supports_h3_direction": directional,
        "note": "H3 full decision also requires >=3/5 rating strata (see strata).",
    }


def cif_bias_by_rating(
    df: pd.DataFrame,
    *,
    horizon_months: float | None = None,
    n_bootstrap: int = 200,
    seed: int = 0,
    min_n: int = 50,
) -> dict[str, Any]:
    """Per-Rating Δ for H3 stratum rule (cfg asks 5 strata; we use all with n≥min_n)."""
    if "Rating" not in df.columns:
        return {
            "metric": "cif_bias_by_rating",
            "status": "error",
            "error": "Rating column missing",
        }
    cr = build_cr_frame(df)
    ratings = (
        cr["Rating"]
        .astype(str)
        .value_counts()
        .loc[lambda s: s >= min_n]
        .index.tolist()
    )
    rows = []
    n_support = 0
    for i, r in enumerate(ratings):
        sub = cr.loc[cr["Rating"].astype(str) == r]
        bias = cif_bias(
            sub["cr_time"].to_numpy(),
            sub["cr_event"].to_numpy(),
            horizon_months=horizon_months,
            n_bootstrap=n_bootstrap,
            seed=seed + i + 1,
        )
        ok = bool(bias.get("supports_h3_direction"))
        n_support += int(ok)
        rows.append({"rating": r, "n": int(len(sub)), **{k: bias[k] for k in (
            "delta", "abs_delta", "supports_h3_direction", "reject_h0_pointwise",
            "bootstrap",
        )}})

    min_strata = int(cfg.PROTOCOL["hypotheses"]["H3"]["min_strata_consistent"])
    return {
        "metric": "cif_bias_by_rating",
        "status": "live",
        "n_strata": len(rows),
        "n_strata_supporting_h3": n_support,
        "min_strata_consistent": min_strata,
        "h3_strata_rule_met": bool(n_support >= min_strata),
        "strata": rows,
    }


def cumulative_incidence(
    time: np.ndarray,
    cr_event: np.ndarray,
    *,
    horizon_months: float | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Bundle naive + AJ at horizon (alias used by E05)."""
    return cif_bias(time, cr_event, horizon_months=horizon_months, n_bootstrap=0)


def cause_specific_eval(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    return {
        "metric": "cause_specific",
        "status": "stub",
        "value": None,
        "note": "Requires separate *_cr Cox fit; do not overwrite Fase A baseline.",
    }


def fine_gray_eval(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    return {
        "metric": "fine_gray",
        "status": "stub",
        "value": None,
        "note": "Requires separate *_cr Fine–Gray fit (F01 / later train).",
    }


def wolbers_concordance(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    return {
        "metric": "wolbers_c",
        "status": "stub",
        "value": None,
        "note": "Needs cause-specific risk predictions from *_cr models.",
    }
