"""
Anchor harness — port of Lillelund et al. ``ladder_hypo.ipynb`` experiment.

Imports ``dgp`` / ``utility`` from the AN00 mirror under
``data/raw/anchor/position-cindex/``. Used by ``AN02_ANCHOR_run_ladder_hypo.py``
to recover Table 2 / Figure 5 style summaries.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from SurvivalEVAL import SurvivalEvaluator
from lifelines import CoxPHFitter
from pycop import simulation
from sksurv.metrics import concordance_index_ipcw

from src.config import cfg


def _author_code_dir() -> Path:
    return Path(cfg.ANCHOR["raw_dir"])


def ensure_author_path() -> Path:
    raw = _author_code_dir()
    if not (raw / "dgp.py").exists():
        raise FileNotFoundError(f"Missing author dgp.py — run AN00 first ({raw})")
    path = str(raw)
    if path not in sys.path:
        sys.path.insert(0, path)
    return raw


def _uv_seed(seed: int, tau: float) -> int:
    return int(seed * 1_000_003 + round(float(tau) * 10_000))


def fit_predict_cox_survival(df_train, df_test, features):
    from utility import lifelines_surv_to_matrix  # noqa: WPS433 — author mirror

    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(df_train[["time", "event"] + features], duration_col="time", event_col="event")
    surv_df = cph.predict_survival_function(df_test[features])
    time_grid, S = lifelines_surv_to_matrix(surv_df)
    risks = cph.predict_log_partial_hazard(df_test[features]).values.reshape(-1)
    return cph, time_grid, S, risks


def eval_ibs_survivaleval(
    survival_outputs,
    time_bins,
    true_test_time,
    true_test_event,
    test_time,
    test_event,
    train_time,
    train_event,
    num_points=10,
):
    survival_df = pd.DataFrame(survival_outputs, columns=time_bins.cpu().numpy())

    true_eval = SurvivalEvaluator(survival_df, time_bins, true_test_time, true_test_event)
    ibs_true = true_eval.integrated_brier_score(IPCW_weighted=False, num_points=num_points)

    eval_obs = SurvivalEvaluator(
        survival_df,
        time_bins,
        test_time,
        test_event,
        train_time,
        train_event,
    )
    ibs_uncens = eval_obs.integrated_brier_score(IPCW_weighted=False, num_points=num_points)
    ibs_ipcw = eval_obs.integrated_brier_score(IPCW_weighted=True, num_points=num_points)
    return ibs_true, ibs_uncens, ibs_ipcw


def eval_ci_survivaleval(
    survival_outputs,
    time_bins,
    true_test_time,
    true_test_event,
    test_time,
    test_event,
    train_time,
    train_event,
    ties="All",
    pair_method="Comparable",
    tau=None,
):
    """Concordance via SurvivalEVAL + sksurv Uno (author notebook path).

    Harrell / oracle: SurvivalEVAL concordance.
    Uno: **sksurv** ``concordance_index_ipcw`` first — matches author
    ``ladder_hypo.ipynb`` / Figure 5. SurvivalEVAL ``method='Uno'`` differs
    numerically on newer package versions.
    """
    from utility import convert_to_structured  # noqa: WPS433

    if hasattr(time_bins, "detach"):
        grid = time_bins.detach().cpu().numpy().astype(float)
    else:
        grid = np.asarray(time_bins, dtype=float)

    survival_df = pd.DataFrame(np.asarray(survival_outputs, dtype=float), columns=grid)

    eval_obs = SurvivalEvaluator(
        survival_df,
        time_bins,
        test_time,
        test_event,
        train_time,
        train_event,
    )
    true_eval = SurvivalEvaluator(survival_df, time_bins, true_test_time, true_test_event)

    def _concordance_harrell(evaluator) -> float:
        try:
            return float(evaluator.concordance(ties=ties, method="Harrell")[0])
        except TypeError:
            try:
                return float(
                    evaluator.concordance(ties=ties, pair_method=pair_method)[0]
                )
            except ValueError:
                return float("nan")
        except ValueError:
            return float("nan")

    ci_harrell = _concordance_harrell(eval_obs)
    ci_true = _concordance_harrell(true_eval)

    # Author Figure 5 Uno path (sksurv IPCW on predicted times)
    try:
        pred_times = np.asarray(eval_obs.predicted_event_times, dtype=float)
        risks = -pred_times
        y_train = convert_to_structured(train_time, train_event.astype(bool))
        y_test = convert_to_structured(test_time, test_event.astype(bool))
        tau_u = float(tau) if tau is not None else float(np.quantile(train_time, 0.8))
        ci_uno = float(concordance_index_ipcw(y_train, y_test, risks, tau=tau_u)[0])
    except Exception:
        ci_uno = float("nan")

    return float(ci_true), float(ci_harrell), float(ci_uno)


SETTING_TO_SCENARIO = {
    "random": "random",
    "indep": "independent",
    0.25: "dep_tau25",
    0.5: "dep_tau50",
    0.75: "dep_tau75",
    "0.25": "dep_tau25",
    "0.5": "dep_tau50",
    "0.75": "dep_tau75",
}


def run_bias_experiment(
    data_cfg: dict[str, Any] | None = None,
    taus: Iterable[float] = (0.25, 0.5, 0.75),
    copula_name: str = "clayton",
    seeds: Iterable[int] = range(5),
    device: str = "cpu",
    dtype=torch.float64,
    num_points: int = 10,
    train_frac: float = 0.7,
    split_seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Faithful port of the author notebook ``run_bias_experiment``."""
    ensure_author_path()
    from dgp import DGP_Weibull_linear  # noqa: WPS433
    from utility import (  # noqa: WPS433
        kendall_tau_to_theta,
        make_time_bins,
        train_test_split_df,
    )

    data_cfg = data_cfg or dict(cfg.ANCHOR["data_cfg"])
    alpha_c = data_cfg["alpha_e1"]
    gamma_c = data_cfg["gamma_e1"]
    alpha_e = data_cfg["alpha_e2"]
    gamma_e = data_cfg["gamma_e2"]
    n_samples = data_cfg["n_samples"]
    n_features = data_cfg["n_features"]

    settings = [
        {"label": "random", "tau": 0.0, "censor_use_x": False, "use_copula": False},
        {"label": "indep", "tau": 0.0, "censor_use_x": True, "use_copula": False},
    ] + [
        {"label": float(t), "tau": float(t), "censor_use_x": True, "use_copula": True}
        for t in taus
    ]

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))
        X = torch.rand((n_samples, n_features), generator=g, device=device, dtype=dtype)
        beta_event = 2 * torch.rand((n_features,), generator=g, device=device, dtype=dtype) - 1
        beta_cens = 2 * torch.rand((n_features,), generator=g, device=device, dtype=dtype) - 1

        dgp_event_base = DGP_Weibull_linear(
            n_features,
            alpha_e,
            gamma_e,
            use_x=True,
            device=device,
            dtype=dtype,
            coeff=beta_event,
        )

        for s in settings:
            label = s["label"]
            tau = s["tau"]
            censor_use_x = s["censor_use_x"]
            use_copula = s["use_copula"]

            if (not use_copula) or (tau == 0.0):
                rng = np.random.default_rng(int(seed))
                u = torch.tensor(rng.uniform(0, 1, n_samples), device=device, dtype=dtype)
                v = torch.tensor(rng.uniform(0, 1, n_samples), device=device, dtype=dtype)
            else:
                np.random.seed(_uv_seed(int(seed), float(tau)))
                theta = kendall_tau_to_theta(copula_name, float(tau))
                u_np, v_np = simulation.simu_archimedean(
                    copula_name, 2, n_samples, theta=theta
                )
                u = torch.from_numpy(u_np).to(device=device, dtype=dtype).reshape(-1)
                v = torch.from_numpy(v_np).to(device=device, dtype=dtype).reshape(-1)

            dgp_cens = DGP_Weibull_linear(
                n_features,
                alpha_c,
                gamma_c,
                use_x=censor_use_x,
                device=device,
                dtype=dtype,
                coeff=beta_cens,
            )

            t_c = dgp_cens.rvs(X, u)
            t_e = dgp_event_base.rvs(X, v)
            T = np.minimum(t_e, t_c)
            E = (t_e < t_c).astype(int)

            df = pd.DataFrame(
                X.detach().cpu().numpy(), columns=[f"X{i}" for i in range(n_features)]
            )
            df["time"] = np.where(T <= 0, 1.0, T)
            df["event"] = E
            df["true_time"] = np.where(t_e <= 0, 1.0, t_e)
            df["true_censor"] = np.where(t_c <= 0, 1.0, t_c)
            df = (
                df.replace([np.inf, -np.inf], np.nan)
                .dropna(subset=["time", "true_time"])
                .reset_index(drop=True)
            )

            df_train, df_test = train_test_split_df(
                df, train_frac=train_frac, seed=int(split_seed)
            )
            features = [c for c in df.columns if c.startswith("X")]

            train_time = df_train["time"].values
            train_event = df_train["event"].values
            test_time = df_test["time"].values
            test_event = df_test["event"].values
            true_test_time = df_test["true_time"].values
            true_test_event = np.ones_like(true_test_time, dtype=int)

            time_grid = make_time_bins(df["true_time"].values, event=None)
            time_bins = torch.tensor(time_grid, device=device, dtype=dtype)

            _, cox_time_grid, S, _ = fit_predict_cox_survival(df_train, df_test, features)
            S_interp = np.vstack(
                [
                    np.interp(time_grid, cox_time_grid, S[i], left=1.0, right=S[i, -1])
                    for i in range(S.shape[0])
                ]
            )

            ibs_true, ibs_uncens, ibs_ipcw = eval_ibs_survivaleval(
                survival_outputs=S_interp,
                time_bins=time_bins,
                true_test_time=true_test_time,
                true_test_event=true_test_event,
                test_time=test_time,
                test_event=test_event,
                train_time=train_time,
                train_event=train_event,
                num_points=num_points,
            )
            ci_true, ci_harrell, ci_uno = eval_ci_survivaleval(
                survival_outputs=S_interp,
                time_bins=time_bins,
                true_test_time=true_test_time,
                true_test_event=true_test_event,
                test_time=test_time,
                test_event=test_event,
                train_time=train_time,
                train_event=train_event,
                tau=None,
            )

            rows.append(
                {
                    "setting": label,
                    "tau": float(tau),
                    "seed": int(seed),
                    "censor_rate": float(1.0 - df["event"].mean()),
                    "n_events": int(df["event"].sum()),
                    "ibs_true": float(ibs_true),
                    "ibs_uncens": float(ibs_uncens),
                    "ibs_ipcw": float(ibs_ipcw),
                    "bias_uncens": float(ibs_uncens - ibs_true),
                    "bias_ipcw": float(ibs_ipcw - ibs_true),
                    "ci_true": float(ci_true),
                    "ci_harrell": float(ci_harrell),
                    "ci_uno": float(ci_uno),
                    "bias_ci_harrell": float(ci_harrell - ci_true),
                    "bias_ci_uno": float(ci_uno - ci_true),
                    "censor_use_x": bool(censor_use_x),
                    "use_copula": bool(use_copula),
                }
            )

    res = pd.DataFrame(rows)
    summary = (
        res.groupby(["setting"], as_index=False)
        .agg(
            tau=("tau", "mean"),
            censor_rate=("censor_rate", "mean"),
            censor_rate_std=("censor_rate", "std"),
            n_events=("n_events", "mean"),
            bias_uncens=("bias_uncens", "mean"),
            bias_ipcw=("bias_ipcw", "mean"),
            bias_uncens_std=("bias_uncens", "std"),
            bias_ipcw_std=("bias_ipcw", "std"),
            ibs_true=("ibs_true", "mean"),
            ibs_uncens=("ibs_uncens", "mean"),
            ibs_ipcw=("ibs_ipcw", "mean"),
            bias_ci_harrell=("bias_ci_harrell", "mean"),
            bias_ci_uno=("bias_ci_uno", "mean"),
            bias_ci_harrell_std=("bias_ci_harrell", "std"),
            bias_ci_uno_std=("bias_ci_uno", "std"),
            ci_true=("ci_true", "mean"),
            ci_harrell=("ci_harrell", "mean"),
            ci_uno=("ci_uno", "mean"),
            ci_true_std=("ci_true", "std"),
            ibs_true_std=("ibs_true", "std"),
        )
    )

    def _sort_key(x):
        if x == "random":
            return -2.0
        if x == "indep":
            return -1.0
        return float(x)

    summary["__k"] = summary["setting"].apply(_sort_key)
    summary = summary.sort_values("__k").drop(columns="__k").reset_index(drop=True)
    return res, summary


def summary_to_scenario_dict(summary: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Map author summary rows → AN03 scenario_summary schema."""
    out: dict[str, dict[str, Any]] = {}
    for _, r in summary.iterrows():
        setting = r["setting"]
        key = SETTING_TO_SCENARIO.get(setting)
        if key is None:
            try:
                key = SETTING_TO_SCENARIO.get(float(setting))
            except (TypeError, ValueError):
                key = None
        if key is None:
            continue
        out[key] = {
            "n_events": float(r["n_events"]),
            "censor_pct": float(r["censor_rate"]) * 100.0,
            "ci_oracle_mean": float(r["ci_true"]),
            "ci_oracle_sd": float(r.get("ci_true_std") or 0.0),
            "ibs_oracle_mean": float(r["ibs_true"]),
            "ibs_oracle_sd": float(r.get("ibs_true_std") or 0.0),
            "ci_harrell_mean": float(r["ci_harrell"]),
            "ci_uno_mean": float(r["ci_uno"]),
            "ibs_ipcw_mean": float(r["ibs_ipcw"]),
            "bias_ci_harrell": float(r["bias_ci_harrell"]),
            "bias_ci_uno": float(r["bias_ci_uno"]),
            "bias_ipcw": float(r["bias_ipcw"]),
            "author_setting": setting
            if isinstance(setting, str)
            else float(setting),
        }
    return out
