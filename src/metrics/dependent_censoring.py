"""
Rung 4 — Dependent-censoring sensitivity (SurvivalEVAL CopulaGraphic).

True dependence is non-identifiable — report a Clayton Kendall-τ sweep
aligned with the anchor scenarios (0 / 0.25 / 0.50 / 0.75). Compare:
  - KM (independent) vs Copula-Graphic marginal S(t)
  - mean predicted S(t|x) vs CG under each τ (when model curves exist)

Metric-level CG-IPCW path (paper F06)
-------------------------------------
Operational definition (fixed before any real-data sweep):
  The Copula-Graphic estimator replaces the Kaplan–Meier estimator of the
  *censoring* survival G(t) inside IPCW weights. Adjusted concordance is
  therefore Uno's C with CG-based Ĝ instead of KM-based Ĝ; adjusted IBS is
  IPCW-IBS with the same substitution. At Kendall τ = 0 (α → 0) both reduce
  to the standard KM-IPCW estimators.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from SurvivalEVAL.Evaluations.Concordance import (
    _finalize_counts,
    _is_before_tau,
    _right_censored_risk_counts,
)
from SurvivalEVAL.NonparametricEstimator.SingleEvent import CopulaGraphic, KaplanMeier
from sksurv.nonparametric import kaplan_meier_estimator

from src.config import cfg

# Anchor-aligned grid (cfg.ANCHOR scenarios with clayton / indep)
DEFAULT_K_TAU = (0.0, 0.25, 0.50, 0.75)

# Fine grid for paper F06 sensitivity sweep
PAPER_TAU_GRID = (
    0.0,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
    0.60,
    0.75,
)


def clayton_alpha_from_kendall(k_tau: float) -> float:
    """Clayton: τ = α/(α+2) ⇒ α = 2τ/(1-τ). τ→0 ⇒ α→0 (near independence)."""
    t = float(k_tau)
    if t <= 0.0:
        return 1e-9
    if t >= 1.0:
        return 1e9
    return 2.0 * t / (1.0 - t)


def _censoring_g_predict(
    train_times: np.ndarray,
    train_events: np.ndarray,
    query_times: np.ndarray,
    *,
    alpha: float | None = None,
    backend: str = "km",
) -> np.ndarray:
    """Predict G(t)=P(C>t). backend='km' uses KM; 'cg' uses CopulaGraphic on censoring."""
    train_times = np.asarray(train_times, dtype=float).ravel()
    cens_ind = (~np.asarray(train_events, dtype=bool)).astype(int)
    query = np.asarray(query_times, dtype=float)
    if backend == "km":
        model = KaplanMeier(train_times, cens_ind.astype(bool))
        return np.asarray(model.predict(query), dtype=float)
    if backend == "cg":
        a = 1e-9 if alpha is None else float(max(alpha, 1e-9))
        model = CopulaGraphic(train_times, cens_ind, alpha=a, type="Clayton")
        return np.asarray(model.predict(query), dtype=float)
    raise ValueError(f"Unknown censoring backend: {backend}")


def uno_c_ipcw(
    risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    *,
    train_time: np.ndarray | None = None,
    train_event: np.ndarray | None = None,
    alpha: float | None = None,
    censoring: str = "km",
    tau: float | None = None,
    ties: str = "Risk",
) -> float:
    """
    Uno's C with IPCW weights from KM or CG censoring survival.

    Definition: Ĝ replaces KM in w_i = 1/Ĝ(T_i) for event anchors (SurvivalEVAL Uno).
    ``censoring='cg'`` uses CopulaGraphic on the censoring margin; ``'km'`` is standard Uno.
    """
    risk = np.asarray(risk, dtype=float).ravel()
    time = np.asarray(time, dtype=float).ravel()
    event = np.asarray(event, dtype=bool).ravel()
    if train_time is None:
        train_time = time
    if train_event is None:
        train_event = event
    train_time = np.asarray(train_time, dtype=float).ravel()
    train_event = np.asarray(train_event, dtype=bool).ravel()

    g = _censoring_g_predict(
        train_time, train_event, time, alpha=alpha, backend=censoring
    )
    observed_anchors = event & _is_before_tau(time, tau)
    if np.any(g[observed_anchors] <= 0):
        raise ValueError("Censoring survival Ĝ ≤ 0 for an observed event anchor.")
    ipcw = np.zeros_like(time, dtype=float)
    ipcw[observed_anchors] = 1.0 / g[observed_anchors]
    counts = _right_censored_risk_counts(
        event,
        time,
        risk,
        tau=tau,
        sample_weights=ipcw,
        anchor_pair_weights=np.square(ipcw),
    )
    c_index, _, _ = _finalize_counts(counts, ties=ties.lower())
    return float(c_index)


def harrell_c(risk: np.ndarray, time: np.ndarray, event: np.ndarray) -> float:
    """Harrell C (no IPCW); used as model-identity sanity check vs numbers.json."""
    from SurvivalEVAL.Evaluations.Concordance import concordance

    pred_times = -np.asarray(risk, dtype=float).ravel()
    c, _, _ = concordance(
        pred_times,
        np.asarray(time, dtype=float).ravel(),
        np.asarray(event, dtype=bool).ravel(),
        method="Harrell",
    )
    return float(c)


def _predict_s_at(
    surv_grid: np.ndarray,
    times_grid: np.ndarray,
    target: float,
) -> np.ndarray:
    """Step/linear interpolate each subject's S(t|x) at ``target``. surv_grid: (T, n)."""
    tg = np.asarray(times_grid, dtype=float).ravel()
    S = np.asarray(surv_grid, dtype=float)
    if S.ndim != 2:
        raise ValueError("surv_grid must be (n_times, n)")
    if S.shape[0] != len(tg):
        if S.shape[1] == len(tg):
            S = S.T
        else:
            raise ValueError(f"surv_grid shape {S.shape} vs times {len(tg)}")
    # right-continuous step: last grid time ≤ target
    idx = int(np.searchsorted(tg, target, side="right") - 1)
    if idx < 0:
        return np.ones(S.shape[1], dtype=float)
    return np.clip(S[idx, :], 0.0, 1.0)


def ipcw_brier_at(
    surv_grid: np.ndarray,
    times_grid: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    target_time: float,
    *,
    train_time: np.ndarray | None = None,
    train_event: np.ndarray | None = None,
    alpha: float | None = None,
    censoring: str = "km",
) -> float:
    """Single-time IPCW Brier with KM or CG Ĝ (SurvivalEVAL single_brier_score logic)."""
    time = np.asarray(time, dtype=float).ravel()
    event = np.asarray(event, dtype=bool).ravel()
    if train_time is None:
        train_time = time
    if train_event is None:
        train_event = event
    preds = _predict_s_at(surv_grid, times_grid, target_time)
    event_before = (time <= target_time) & event
    event_free = (time > target_time) | ((time == target_time) & ~event)

    g_obs = _censoring_g_predict(
        train_time, train_event, time, alpha=alpha, backend=censoring
    )
    g_obs = np.asarray(g_obs, dtype=float)
    g_obs[g_obs == 0] = np.inf
    g_t = float(
        _censoring_g_predict(
            train_time,
            train_event,
            np.asarray([target_time], dtype=float),
            alpha=alpha,
            backend=censoring,
        )[0]
    )
    if g_t == 0:
        g_t = np.inf
    w1 = event_before / g_obs
    w1[np.isnan(w1)] = 0.0
    w2 = event_free / g_t
    w2[np.isnan(w2)] = 0.0
    return float(np.mean(np.square(preds) * w1 + np.square(1.0 - preds) * w2))


def ipcw_ibs(
    surv_grid: np.ndarray,
    times_grid: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    *,
    train_time: np.ndarray | None = None,
    train_event: np.ndarray | None = None,
    alpha: float | None = None,
    censoring: str = "km",
    num_points: int = 10,
) -> float:
    """Trapezoidal IPCW-IBS over ``num_points`` times in (0, max observed]."""
    time = np.asarray(time, dtype=float).ravel()
    t_max = float(np.max(time))
    targets = np.linspace(0.0, t_max, int(num_points) + 1)[1:]  # exclude 0
    scores = [
        ipcw_brier_at(
            surv_grid,
            times_grid,
            time,
            event,
            float(t),
            train_time=train_time,
            train_event=train_event,
            alpha=alpha,
            censoring=censoring,
        )
        for t in targets
    ]
    return float(np.trapezoid(scores, targets) / t_max)


def metrics_at_tau(
    risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    surv_grid: np.ndarray,
    times_grid: np.ndarray,
    k_tau: float,
    *,
    num_points: int = 10,
) -> dict[str, float]:
    """Point estimates of CG-IPCW Uno C and IPCW-IBS at one Kendall τ."""
    alpha = clayton_alpha_from_kendall(k_tau)
    return {
        "k_tau": float(k_tau),
        "clayton_alpha": float(alpha),
        "uno_c_cg": uno_c_ipcw(risk, time, event, alpha=alpha, censoring="cg"),
        "ibs_cg": ipcw_ibs(
            surv_grid,
            times_grid,
            time,
            event,
            alpha=alpha,
            censoring="cg",
            num_points=num_points,
        ),
    }


def sanity_tau0(
    risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    surv_grid: np.ndarray,
    times_grid: np.ndarray,
    *,
    harrell_ref: float | None = None,
    ibs_ref: float | None = None,
    atol_bridge: float = 5e-3,
    atol_harrell: float = 1e-3,
    num_points: int = 10,
) -> dict[str, Any]:
    """
    Automatic τ=0 bridge check: CG-IPCW must match KM-IPCW on the same inputs.
    Also checks Harrell against numbers.json when ``harrell_ref`` is given (model ID).
    """
    uno_km = uno_c_ipcw(risk, time, event, censoring="km")
    uno_cg0 = uno_c_ipcw(
        risk, time, event, alpha=clayton_alpha_from_kendall(0.0), censoring="cg"
    )
    ibs_km = ipcw_ibs(
        surv_grid, times_grid, time, event, censoring="km", num_points=num_points
    )
    ibs_cg0 = ipcw_ibs(
        surv_grid,
        times_grid,
        time,
        event,
        alpha=clayton_alpha_from_kendall(0.0),
        censoring="cg",
        num_points=num_points,
    )
    h = harrell_c(risk, time, event)
    out: dict[str, Any] = {
        "harrell": h,
        "uno_km": uno_km,
        "uno_cg_tau0": uno_cg0,
        "ibs_km": ibs_km,
        "ibs_cg_tau0": ibs_cg0,
        "delta_uno_cg_minus_km": float(uno_cg0 - uno_km),
        "delta_ibs_cg_minus_km": float(ibs_cg0 - ibs_km),
        "bridge_ok": bool(
            abs(uno_cg0 - uno_km) <= atol_bridge and abs(ibs_cg0 - ibs_km) <= atol_bridge
        ),
    }
    if harrell_ref is not None:
        out["harrell_ref"] = float(harrell_ref)
        out["harrell_ok"] = bool(abs(h - float(harrell_ref)) <= atol_harrell)
    if ibs_ref is not None:
        out["ibs_ref"] = float(ibs_ref)
        # soft: same backend family; warn-level only (SEVAL vs sksurv may differ)
        out["ibs_ref_delta"] = float(ibs_km - float(ibs_ref))
    if not out["bridge_ok"]:
        raise AssertionError(
            "τ=0 CG↔KM bridge failed: "
            f"ΔUno={out['delta_uno_cg_minus_km']:.6g}, "
            f"ΔIBS={out['delta_ibs_cg_minus_km']:.6g} "
            f"(atol={atol_bridge})"
        )
    if harrell_ref is not None and not out.get("harrell_ok", True):
        raise AssertionError(
            f"Harrell model-identity check failed: got {h:.6f}, ref {harrell_ref:.6f}"
        )
    return out


def bootstrap_metric_sweep(
    risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    surv_grid: np.ndarray,
    times_grid: np.ndarray,
    tau_grid: tuple[float, ...] | list[float] = PAPER_TAU_GRID,
    *,
    B: int = 200,
    seed: int = 42,
    num_points: int = 10,
) -> dict[str, Any]:
    """
    Sweep CG-IPCW Uno C and IBS over Kendall τ with stratified bootstrap CIs.

    Point estimates use the full sample; CIs resample subjects (stratified by event).
    """
    risk = np.asarray(risk, dtype=float).ravel()
    time = np.asarray(time, dtype=float).ravel()
    event = np.asarray(event, dtype=bool).ravel()
    S = np.asarray(surv_grid, dtype=float)
    tg = np.asarray(times_grid, dtype=float).ravel()
    n = len(time)
    tau_grid = tuple(float(t) for t in tau_grid)
    rng = np.random.default_rng(seed)

    idx_e = np.flatnonzero(event)
    idx_c = np.flatnonzero(~event)

    means_c: list[float] = []
    means_ibs: list[float] = []
    lo_c: list[float] = []
    hi_c: list[float] = []
    lo_ibs: list[float] = []
    hi_ibs: list[float] = []
    alphas: list[float] = []

    for k_tau in tau_grid:
        pt = metrics_at_tau(
            risk, time, event, S, tg, k_tau, num_points=num_points
        )
        alphas.append(pt["clayton_alpha"])
        means_c.append(pt["uno_c_cg"])
        means_ibs.append(pt["ibs_cg"])

        boot_c = np.empty(B, dtype=float)
        boot_ibs = np.empty(B, dtype=float)
        for b in range(B):
            take_e = rng.choice(idx_e, size=len(idx_e), replace=True)
            take_c = rng.choice(idx_c, size=len(idx_c), replace=True)
            idx = np.concatenate([take_e, take_c])
            boot_c[b] = uno_c_ipcw(
                risk[idx],
                time[idx],
                event[idx],
                alpha=pt["clayton_alpha"],
                censoring="cg",
            )
            boot_ibs[b] = ipcw_ibs(
                S[:, idx],
                tg,
                time[idx],
                event[idx],
                alpha=pt["clayton_alpha"],
                censoring="cg",
                num_points=num_points,
            )
        lo_c.append(float(np.quantile(boot_c, 0.025)))
        hi_c.append(float(np.quantile(boot_c, 0.975)))
        lo_ibs.append(float(np.quantile(boot_ibs, 0.025)))
        hi_ibs.append(float(np.quantile(boot_ibs, 0.975)))

    return {
        "tau_grid": list(tau_grid),
        "clayton_alpha": alphas,
        "uno_c_adjusted": {
            "mean": means_c,
            "ci_lo": lo_c,
            "ci_hi": hi_c,
        },
        "ibs_adjusted": {
            "mean": means_ibs,
            "ci_lo": lo_ibs,
            "ci_hi": hi_ibs,
        },
        "definition": (
            "CG replaces KM in the censoring survival G(t) used for IPCW weights; "
            "Uno C and IPCW-IBS are recomputed under that Ĝ at each Kendall τ. "
            "Clayton α = 2τ/(1-τ)."
        ),
        "bootstrap": {"B": int(B), "seed": int(seed), "stratified_by_event": True},
        "n": int(n),
        "n_events": int(event.sum()),
    }


def _km_curve(time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tt, surv = kaplan_meier_estimator(event.astype(bool), time.astype(float))
    return np.asarray(tt, dtype=float), np.asarray(surv, dtype=float)


def _step_interp(query: np.ndarray, times: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Right-continuous step: value at last time ≤ t (KM / CG style)."""
    idx = np.searchsorted(times, query, side="right") - 1
    out = np.empty_like(query, dtype=float)
    out[idx < 0] = 1.0
    ok = idx >= 0
    out[ok] = values[idx[ok]]
    return out


def _eval_grid(time: np.ndarray, n_grid: int = 40) -> np.ndarray:
    lo = 0.0
    hi = float(np.quantile(time, 0.95))
    if not np.isfinite(hi) or hi <= 0:
        hi = float(np.max(time))
    grid = np.linspace(lo, hi, n_grid)
    grid[0] = 0.0
    return grid


def copula_sensitivity_sweep(
    time: np.ndarray | None = None,
    event: np.ndarray | None = None,
    *,
    surv_grid: np.ndarray | None = None,
    times_grid: np.ndarray | None = None,
    k_taus: tuple[float, ...] | list[float] | None = None,
    copula: str = "Clayton",
    n_grid: int = 40,
    **_kwargs: Any,
) -> dict[str, Any]:
    """
    Sweep CopulaGraphic vs KM (and vs mean predicted S if provided).

    Parameters
    ----------
    time, event:
        Observed follow-up on the eval set.
    surv_grid, times_grid:
        Optional model S(t|x); ``surv_grid`` as (n_times, n) or (n, n_times).
    """
    if time is None or event is None:
        return {
            "metric": "copula_graphic_sweep",
            "status": "stub",
            "limitation": "true_dependence_non_identifiable",
            "grid": None,
            "curve": None,
        }

    time = np.asarray(time, dtype=float).ravel()
    event = np.asarray(event).astype(int).ravel()
    k_taus = tuple(DEFAULT_K_TAU if k_taus is None else k_taus)
    copula = str(copula)
    # Prefer anchor default
    if not copula:
        copula = str(cfg.ANCHOR.get("copula") or "Clayton").capitalize()
        if copula.lower() == "clayton":
            copula = "Clayton"

    grid_t = _eval_grid(time, n_grid=n_grid)
    km_t, km_s = _km_curve(time, event)
    km_on_grid = _step_interp(grid_t, km_t, km_s)

    mean_pred_on_grid = None
    if surv_grid is not None and times_grid is not None:
        times_grid = np.asarray(times_grid, dtype=float).ravel()
        S = np.asarray(surv_grid, dtype=float)
        if S.shape[0] == len(times_grid) and S.shape[1] != len(times_grid):
            mean_curve = S.mean(axis=1)
            mean_pred_on_grid = _step_interp(grid_t, times_grid, mean_curve)
        elif S.shape[1] == len(times_grid):
            mean_curve = S.mean(axis=0)
            mean_pred_on_grid = _step_interp(grid_t, times_grid, mean_curve)

    rows: list[dict[str, Any]] = []
    for k_tau in k_taus:
        alpha = clayton_alpha_from_kendall(k_tau)
        try:
            cg = CopulaGraphic(time, event, alpha=alpha, type=copula)
            cg_on_grid = np.asarray(cg.predict(grid_t), dtype=float).ravel()
            mae_vs_km = float(np.mean(np.abs(cg_on_grid - km_on_grid)))
            row: dict[str, Any] = {
                "k_tau": float(k_tau),
                "clayton_alpha": float(alpha),
                "copula": copula,
                "status": "live",
                "mae_cg_vs_km": mae_vs_km,
                "median_cg_s": float(np.median(cg_on_grid)),
                "median_km_s": float(np.median(km_on_grid)),
                # compact curve (every other point)
                "curve": [
                    {
                        "t": float(grid_t[i]),
                        "km": float(km_on_grid[i]),
                        "cg": float(cg_on_grid[i]),
                    }
                    for i in range(0, len(grid_t), max(1, len(grid_t) // 10))
                ],
            }
            if mean_pred_on_grid is not None:
                mae_pred = float(np.mean(np.abs(mean_pred_on_grid - cg_on_grid)))
                mae_pred_km = float(np.mean(np.abs(mean_pred_on_grid - km_on_grid)))
                row["mae_mean_pred_vs_cg"] = mae_pred
                row["mae_mean_pred_vs_km"] = mae_pred_km
                row["delta_mae_pred_cg_minus_km"] = mae_pred - mae_pred_km
            rows.append(row)
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "k_tau": float(k_tau),
                    "clayton_alpha": float(alpha),
                    "copula": copula,
                    "status": "error",
                    "error": str(exc),
                }
            )

    # Sensitivity summary: how much CG drifts from KM as τ increases
    live = [r for r in rows if r.get("status") == "live"]
    mae_by_tau = {str(r["k_tau"]): r.get("mae_cg_vs_km") for r in live}
    max_mae = max((r["mae_cg_vs_km"] for r in live), default=None)

    return {
        "metric": "copula_graphic_sweep",
        "status": "live" if live else "error",
        "backend": "SurvivalEVAL.CopulaGraphic",
        "limitation": "true_dependence_non_identifiable",
        "note": (
            "Clayton Kendall-τ grid matches anchor AN scenarios. "
            "CG vs KM = how much the marginal event survival would shift "
            "if censoring were dependent at that τ; not a test that τ is true."
        ),
        "copula": copula,
        "k_taus": list(k_taus),
        "n": int(len(time)),
        "n_events": int(event.sum()),
        "n_grid": int(len(grid_t)),
        "mae_cg_vs_km_by_tau": mae_by_tau,
        "max_mae_cg_vs_km": max_mae,
        "grid": rows,
    }
