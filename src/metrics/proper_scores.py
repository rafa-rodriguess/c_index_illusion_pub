"""
Rung 3 — Proper scores via SurvivalEVAL (anchor stack).

Primary H1 path: IPCW Integrated Brier Score.
Also: naive (non-IPCW) IBS + pointwise Brier at configured horizons.
CRPS / censored log-likelihood: stub (not exposed on SurvivalEvaluator 0.8).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from src.config import cfg

if TYPE_CHECKING:
    from SurvivalEVAL import SurvivalEvaluator


def _survival_evaluator_type():
    """Lazy import — SurvivalEVAL before XGB Booster load SIGSEGVs on this stack."""
    from SurvivalEVAL import SurvivalEvaluator

    return SurvivalEvaluator

DAYS_PER_MONTH = 365.25 / 12.0


def _unwrap(score: Any) -> float:
    if isinstance(score, tuple):
        score = score[0]
    return float(score)


def horizon_times_days(
    horizons_months: list[int] | None = None,
    *,
    unit: str = "days",
) -> dict[str, float]:
    """Map configured month horizons to evaluation times in domain units."""
    months = list(horizons_months or cfg.EVAL["ibs_horizons_months"])
    out: dict[str, float] = {}
    for m in months:
        if unit == "days":
            out[str(m)] = float(m) * DAYS_PER_MONTH
        elif unit == "months":
            out[str(m)] = float(m)
        else:
            raise ValueError(f"Unknown time unit: {unit}")
    return out


def time_unit_for_domain(domain_id: str) -> str:
    # D1 Backblaze / D2 Bondora: duration in days. D3 Stack Exchange: months.
    if domain_id == "DOMAIN_03":
        return "months"
    return "days"


def ipcw_integrated_brier(
    evaluator: Any,
    *,
    num_points: int = 10,
    **_kwargs: Any,
) -> dict[str, Any]:
    """SurvivalEVAL IBS with IPCW (anchor Figura 5 path)."""
    _survival_evaluator_type()  # ensure package available
    try:
        value = _unwrap(
            evaluator.integrated_brier_score(
                num_points=num_points,
                IPCW_weighted=True,
                draw_figure=False,
            )
        )
        return {
            "metric": "ipcw_ibs",
            "status": "live",
            "backend": "SurvivalEVAL.SurvivalEvaluator.integrated_brier_score",
            "value": value,
            "num_points": num_points,
            "IPCW_weighted": True,
            "method": "survivaleval_ipcw_ibs",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "metric": "ipcw_ibs",
            "status": "error",
            "backend": "SurvivalEVAL.SurvivalEvaluator.integrated_brier_score",
            "value": None,
            "IPCW_weighted": True,
            "error": str(exc),
        }


def naive_integrated_brier(
    evaluator: Any,
    *,
    num_points: int = 10,
    **_kwargs: Any,
) -> dict[str, Any]:
    """SurvivalEVAL IBS without IPCW (Figura 5 naive counterpart)."""
    try:
        value = _unwrap(
            evaluator.integrated_brier_score(
                num_points=num_points,
                IPCW_weighted=False,
                draw_figure=False,
            )
        )
        return {
            "metric": "naive_ibs",
            "status": "live",
            "backend": "SurvivalEVAL.SurvivalEvaluator.integrated_brier_score",
            "value": value,
            "num_points": num_points,
            "IPCW_weighted": False,
            "method": "survivaleval_naive_ibs",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "metric": "naive_ibs",
            "status": "error",
            "value": None,
            "IPCW_weighted": False,
            "error": str(exc),
        }


def brier_at_horizons(
    evaluator: Any,
    *,
    domain_id: str,
    horizons_months: list[int] | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Pointwise IPCW Brier at cfg month horizons (converted to domain time unit)."""
    unit = time_unit_for_domain(domain_id)
    targets = horizon_times_days(horizons_months, unit=unit)
    t_max = float(np.max(evaluator.event_times))
    values: dict[str, Any] = {}
    details: dict[str, Any] = {}
    for label, t in targets.items():
        if t > t_max:
            values[label] = None
            details[label] = {
                "target_time": t,
                "status": "skipped_beyond_followup",
                "t_max": t_max,
            }
            continue
        try:
            bs = _unwrap(evaluator.brier_score(target_time=t, IPCW_weighted=True))
            values[label] = bs
            details[label] = {
                "target_time": t,
                "status": "live",
                "IPCW_weighted": True,
            }
        except Exception as exc:  # noqa: BLE001
            values[label] = None
            details[label] = {
                "target_time": t,
                "status": "error",
                "error": str(exc),
            }

    n_live = sum(1 for v in values.values() if v is not None)
    return {
        "metric": "brier_at_horizons",
        "status": "live" if n_live else "error",
        "backend": "SurvivalEVAL.SurvivalEvaluator.brier_score",
        "time_unit": unit,
        "horizons_months": list(horizons_months or cfg.EVAL["ibs_horizons_months"]),
        "values": values,
        "details": details,
    }


def ipcw_integrated_brier_sksurv(
    *,
    time_train: np.ndarray,
    event_train: np.ndarray,
    time_test: np.ndarray,
    event_test: np.ndarray,
    surv_grid: np.ndarray,
    times_grid: np.ndarray,
    q_low: float = 0.05,
    q_high: float = 0.80,
) -> dict[str, Any]:
    """
    IPCW IBS via sksurv (stable fallback when SurvivalEVAL segfaults / fails).

    ``surv_grid`` is (n_times, n); clipped to (0,1]. Integration grid is the
    subset of ``times_grid`` strictly inside the test follow-up percentiles.
    """
    from sksurv.metrics import integrated_brier_score
    from sksurv.util import Surv

    t_te = np.asarray(time_test, dtype=float)
    lo = float(np.quantile(t_te, q_low))
    hi = float(np.quantile(t_te, q_high))
    tg = np.asarray(times_grid, dtype=float)
    mask = (tg > lo) & (tg < hi) & (tg > float(t_te.min())) & (tg < float(t_te.max()))
    if int(mask.sum()) < 3:
        # fallback denser grid inside follow-up
        tg_use = np.linspace(max(lo, float(t_te.min()) + 1e-6), min(hi, float(t_te.max()) - 1e-6), 40)
        # interpolate S to tg_use
        S_full = np.clip(np.asarray(surv_grid, dtype=float), 1e-15, 1.0)
        S_use = np.vstack(
            [
                np.array(
                    [
                        float(np.interp(t, tg, S_full[:, j], left=1.0, right=float(S_full[-1, j])))
                        for j in range(S_full.shape[1])
                    ]
                )
                for t in tg_use
            ]
        )
    else:
        tg_use = tg[mask]
        S_use = np.clip(np.asarray(surv_grid, dtype=float)[mask, :], 1e-15, 1.0)

    y_tr = Surv.from_arrays(np.asarray(event_train, dtype=bool), np.asarray(time_train, dtype=float))
    y_te = Surv.from_arrays(np.asarray(event_test, dtype=bool), t_te)
    try:
        value = float(integrated_brier_score(y_tr, y_te, S_use.T, tg_use))
        return {
            "metric": "ipcw_ibs",
            "status": "live",
            "backend": "sksurv.metrics.integrated_brier_score",
            "value": value,
            "n_times": int(len(tg_use)),
            "t_low": float(tg_use[0]),
            "t_high": float(tg_use[-1]),
            "method": "sksurv_ipcw_ibs",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "metric": "ipcw_ibs",
            "status": "error",
            "backend": "sksurv.metrics.integrated_brier_score",
            "value": None,
            "error": str(exc),
        }


def censored_loglik(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    return {
        "metric": "censored_loglik",
        "status": "stub",
        "value": None,
        "note": "Not exposed on SurvivalEVAL SurvivalEvaluator 0.8; deferred.",
    }


def crps(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    return {
        "metric": "crps",
        "status": "stub",
        "value": None,
        "note": "Not exposed on SurvivalEVAL SurvivalEvaluator 0.8; deferred.",
    }
