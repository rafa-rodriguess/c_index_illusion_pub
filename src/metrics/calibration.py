"""
Rung 2 — Calibration via SurvivalEVAL (same stack as the anchor harness).

Primary H2 path: ``SurvivalEvaluator.d_calibration`` (Haider / SurvivalEVAL).
Also: ``integrated_calibration_index`` (Austin) and ``km_calibration``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.config import cfg


def make_survival_evaluator(
    surv_grid: np.ndarray,
    times_grid: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    train_time: np.ndarray | None = None,
    train_event: np.ndarray | None = None,
) -> Any:
    """
    Build SurvivalEVAL evaluator.

    ``surv_grid`` may be (n_times, n) or (n, n_times); we normalize to
    subjects × times, matching ``utility.lifelines_surv_to_matrix`` / AN harness.

    SurvivalEVAL is imported lazily: loading it before an XGB Booster
    ``joblib.load`` SIGSEGVs on this macOS/Python stack.
    """
    from SurvivalEVAL import SurvivalEvaluator

    times_grid = np.asarray(times_grid, dtype=float).ravel()
    surv_grid = np.asarray(surv_grid, dtype=float)
    if surv_grid.shape[0] == len(times_grid) and surv_grid.shape[1] != len(times_grid):
        # (n_times, n) → (n, n_times)
        S = surv_grid.T
    elif surv_grid.shape[1] == len(times_grid):
        S = surv_grid
    else:
        raise ValueError(
            f"surv_grid shape {surv_grid.shape} incompatible with "
            f"times_grid length {len(times_grid)}"
        )

    survival_df = pd.DataFrame(S, columns=times_grid)
    kwargs: dict[str, Any] = {}
    if train_time is not None and train_event is not None:
        kwargs["train_event_times"] = np.asarray(train_time, dtype=float)
        kwargs["train_event_indicators"] = np.asarray(train_event, dtype=int)

    return SurvivalEvaluator(
        survival_df,
        times_grid,
        np.asarray(time, dtype=float),
        np.asarray(event, dtype=int),
        **kwargs,
    )


def d_calibration(
    evaluator: SurvivalEvaluator,
    n_bins: int | None = None,
    alpha: float | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    """SurvivalEVAL ``SurvivalEvaluator.d_calibration`` (Haider binning on S(t_i|x_i))."""
    n_bins = int(n_bins or cfg.EVAL["d_calibration_bins"])
    alpha = float(alpha if alpha is not None else cfg.EVAL["d_calibration_alpha"])

    p_value, details = evaluator.d_calibration(num_bins=n_bins, return_details=True)
    # Drop matplotlib artists from serializable payload
    for key in ("histogram_plot", "pp_plot"):
        pair = details.get(key)
        if pair is not None:
            fig = pair[0]
            plt.close(fig)

    p_value = float(p_value)
    hist = np.asarray(details["histogram"], dtype=float)
    return {
        "metric": "d_calibration",
        "status": "live",
        "backend": "SurvivalEVAL.SurvivalEvaluator.d_calibration",
        "survivaleval_version": _se_version(),
        "n_bins": n_bins,
        "alpha": alpha,
        "chi2": float(details["statistics"]),
        "p_value": p_value,
        "reject_h0_well_calibrated": bool(p_value < alpha),
        "bin_counts": hist.tolist(),
        "x_calibration": float(details["x_calibration"]),
        "n": int(len(evaluator.event_times)),
        "n_events": int(np.asarray(evaluator.event_indicators).sum()),
        "method": "haider_via_survivaleval",
    }


def integrated_calibration_index(
    evaluator: SurvivalEvaluator,
    horizon: float | None = None,
    knots: int = 3,
    eps: float = 1e-6,
    **_kwargs: Any,
) -> dict[str, Any]:
    """
    SurvivalEVAL Austin ICI at horizon τ (default: median observed time).

    Event probabilities are clipped to (eps, 1-eps) before the CLL transform —
    SurvivalEVAL's ``log(-log(1-p))`` is undefined at p∈{0,1}, common with
    strong Cox discrimination (e.g. Backblaze).
    """
    from SurvivalEVAL.Evaluations.SingleTimeCalibration import (
        integrated_calibration_index as se_ici,
    )

    if horizon is None:
        horizon = float(np.median(evaluator.event_times))
    try:
        # S(τ|x); SurvivalEVAL ICI wants event probability 1-S
        surv_at_tau = np.asarray(
            evaluator.predict_probability_from_curve(horizon), dtype=float
        ).ravel()
        event_probs = np.clip(1.0 - surv_at_tau, eps, 1.0 - eps)
        summary = se_ici(
            preds=event_probs,
            event_time=evaluator.event_times,
            event_indicator=evaluator.event_indicators,
            target_time=horizon,
            knots=knots,
            draw_figure=False,
        )
        if isinstance(summary, tuple):
            summary = summary[0]
        curve = summary.get("curve") or {}
        return {
            "metric": "ici",
            "status": "live",
            "backend": "SurvivalEVAL.Evaluations.integrated_calibration_index",
            "value": float(summary["ICI"]),
            "E50": float(summary["E50"]),
            "E90": float(summary["E90"]),
            "E_max": float(summary["E_max"]),
            "horizon": float(horizon),
            "knots": knots,
            "event_prob_clip_eps": eps,
            "method": "austin_2020_spline_cox_survivaleval",
            "curve_n_grid": int(len(curve["grid"])) if "grid" in curve else None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "metric": "ici",
            "status": "error",
            "backend": "SurvivalEVAL.Evaluations.integrated_calibration_index",
            "value": None,
            "horizon": float(horizon) if horizon is not None else None,
            "error": str(exc),
        }


def km_vs_predicted(
    evaluator: SurvivalEvaluator,
    **_kwargs: Any,
) -> dict[str, Any]:
    """SurvivalEVAL ``km_calibration`` (ISE between mean predicted S and KM)."""
    try:
        score = evaluator.km_calibration(draw_figure=False)
        if isinstance(score, tuple):
            score = score[0]
        return {
            "metric": "km_vs_predicted",
            "status": "live",
            "backend": "SurvivalEVAL.SurvivalEvaluator.km_calibration",
            "value": float(score),
            "mae": None,  # SurvivalEVAL reports integrated squared error, not MAE
            "km_calibration_ise": float(score),
            "method": "survivaleval_km_calibration",
            "note": "Lower is better; normalized integrated squared error (not MAE).",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "metric": "km_vs_predicted",
            "status": "error",
            "backend": "SurvivalEVAL.SurvivalEvaluator.km_calibration",
            "value": None,
            "mae": None,
            "error": str(exc),
        }


def _se_version() -> str:
    try:
        import SurvivalEVAL as se

        return str(getattr(se, "__version__", "unknown"))
    except Exception:  # noqa: BLE001
        return "unknown"
