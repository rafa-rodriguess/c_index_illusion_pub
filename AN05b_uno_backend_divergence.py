"""
AN05b_uno_backend_divergence.py — SurvivalEVAL vs sksurv Uno C under τ=0.50
===========================================================================
Quantifies the library divergence described qualitatively in §5.2.

For dependent censoring at Kendall τ = 0.50, re-runs the anchor synthetic
Cox ladder (same DGP as AN02) and computes Uno's C two ways on identical
predictions:
  - sksurv ``concordance_index_ipcw`` (author / harness path)
  - SurvivalEVAL ``concordance(method='Uno')``

Writes:
  results/reproduction/ANCHOR_uno_backend_divergence.{json,md}

Execute:
    python -W default AN05b_uno_backend_divergence.py
    python -W default AN05b_uno_backend_divergence.py --n-seeds 100
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.anchor_harness import (  # noqa: E402
    _uv_seed,
    ensure_author_path,
    eval_ci_survivaleval,
    fit_predict_cox_survival,
)
from src.config import cfg
from src.metrics.io import utc_now, write_json

warnings.filterwarnings("default")


def log(msg: str = "") -> None:
    print(msg, flush=True)


def _uno_survivaleval(survival_outputs, time_bins, test_time, test_event, train_time, train_event) -> float:
    from SurvivalEVAL import SurvivalEvaluator

    if hasattr(time_bins, "detach"):
        grid = time_bins.detach().cpu().numpy().astype(float)
    else:
        grid = np.asarray(time_bins, dtype=float)
    survival_df = pd.DataFrame(np.asarray(survival_outputs, dtype=float), columns=grid)
    eval_obs = SurvivalEvaluator(
        survival_df, time_bins, test_time, test_event, train_time, train_event
    )
    return float(eval_obs.concordance(ties="All", method="Uno")[0])


def run_tau50_dual(*, n_seeds: int, tau: float, seed0: int) -> pd.DataFrame:
    ensure_author_path()
    from dgp import DGP_Weibull_linear  # noqa: WPS433
    from pycop import simulation  # noqa: WPS433
    from utility import (  # noqa: WPS433
        kendall_tau_to_theta,
        make_time_bins,
        train_test_split_df,
    )

    data_cfg = dict(cfg.ANCHOR["data_cfg"])
    device = "cpu"
    dtype = torch.float64
    n_samples = data_cfg["n_samples"]
    n_features = data_cfg["n_features"]
    alpha_e, gamma_e = data_cfg["alpha_e2"], data_cfg["gamma_e2"]
    alpha_c, gamma_c = data_cfg["alpha_e1"], data_cfg["gamma_e1"]
    copula_name = "clayton"
    theta = kendall_tau_to_theta(copula_name, float(tau))

    rows: list[dict] = []
    for seed in range(seed0, seed0 + n_seeds):
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))
        X = torch.rand((n_samples, n_features), generator=g, device=device, dtype=dtype)
        beta_event = 2 * torch.rand((n_features,), generator=g, device=device, dtype=dtype) - 1
        beta_cens = 2 * torch.rand((n_features,), generator=g, device=device, dtype=dtype) - 1

        dgp_event = DGP_Weibull_linear(
            n_features, alpha_e, gamma_e, use_x=True, device=device, dtype=dtype, coeff=beta_event
        )
        dgp_cens = DGP_Weibull_linear(
            n_features, alpha_c, gamma_c, use_x=True, device=device, dtype=dtype, coeff=beta_cens
        )

        np.random.seed(_uv_seed(int(seed), float(tau)))
        u_np, v_np = simulation.simu_archimedean(copula_name, 2, n_samples, theta=theta)
        u = torch.from_numpy(u_np).to(device=device, dtype=dtype).reshape(-1)
        v = torch.from_numpy(v_np).to(device=device, dtype=dtype).reshape(-1)

        t_c = dgp_cens.rvs(X, u)
        t_e = dgp_event.rvs(X, v)
        T = np.minimum(t_e, t_c)
        E = (t_e < t_c).astype(int)

        df = pd.DataFrame(X.detach().cpu().numpy(), columns=[f"X{i}" for i in range(n_features)])
        df["time"] = np.where(T <= 0, 1.0, T)
        df["event"] = E
        df["true_time"] = np.where(t_e <= 0, 1.0, t_e)
        df = (
            df.replace([np.inf, -np.inf], np.nan)
            .dropna(subset=["time", "true_time"])
            .reset_index(drop=True)
        )
        df_train, df_test = train_test_split_df(df, train_frac=0.7, seed=0)
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

        ci_true, _ci_h, ci_uno_sk = eval_ci_survivaleval(
            S_interp,
            time_bins,
            true_test_time,
            true_test_event,
            test_time,
            test_event,
            train_time,
            train_event,
            tau=None,
        )
        try:
            ci_uno_se = _uno_survivaleval(
                S_interp, time_bins, test_time, test_event, train_time, train_event
            )
        except Exception as exc:
            log(f"  seed {seed}: SurvivalEVAL Uno failed ({exc})")
            ci_uno_se = float("nan")

        rows.append(
            {
                "seed": int(seed),
                "tau": float(tau),
                "ci_true": float(ci_true),
                "ci_uno_sksurv": float(ci_uno_sk),
                "ci_uno_survivaleval": float(ci_uno_se),
                "bias_sksurv": float(ci_uno_sk - ci_true),
                "bias_survivaleval": float(ci_uno_se - ci_true),
                "delta_ci": float(ci_uno_se - ci_uno_sk),
            }
        )
        if (seed - seed0 + 1) % 10 == 0:
            log(f"  … {seed - seed0 + 1}/{n_seeds} seeds")

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=100)
    parser.add_argument("--tau", type=float, default=0.50)
    parser.add_argument("--seed0", type=int, default=int(cfg.RANDOM_SEED))
    args = parser.parse_args()

    log(f"AN05b: dual Uno at τ={args.tau}, n_seeds={args.n_seeds}, seed0={args.seed0}")
    df = run_tau50_dual(n_seeds=args.n_seeds, tau=args.tau, seed0=args.seed0)
    df = df.dropna(subset=["ci_uno_survivaleval"])

    payload = {
        "stage": "AN05b",
        "tau": args.tau,
        "n_seeds": int(len(df)),
        "seed0": args.seed0,
        "ci_uno_sksurv_mean": float(df["ci_uno_sksurv"].mean()),
        "ci_uno_survivaleval_mean": float(df["ci_uno_survivaleval"].mean()),
        "bias_uno_sksurv_mean": float(df["bias_sksurv"].mean()),
        "bias_uno_survivaleval_mean": float(df["bias_survivaleval"].mean()),
        "delta_ci_uno_mean": float(df["delta_ci"].mean()),
        "delta_bias_mean": float(
            df["bias_survivaleval"].mean() - df["bias_sksurv"].mean()
        ),
        "generated_at_utc": utc_now(),
    }

    out_dir = cfg.DIRS["reproduction"]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "ANCHOR_uno_backend_divergence.json", payload)
    df.to_csv(out_dir / "ANCHOR_uno_backend_divergence_seeds.csv", index=False)

    md = "\n".join(
        [
            f"# AN05b — Uno C backend divergence (τ = {args.tau})",
            "",
            f"n_seeds = {payload['n_seeds']}",
            "",
            f"- Uno C (sksurv IPCW) mean = **{payload['ci_uno_sksurv_mean']:.4f}**",
            f"- Uno C (SurvivalEVAL) mean = **{payload['ci_uno_survivaleval_mean']:.4f}**",
            f"- Δ (SurvivalEVAL − sksurv) = **{payload['delta_ci_uno_mean']:+.4f}**",
            f"- Bias (sksurv − oracle) mean = {payload['bias_uno_sksurv_mean']:+.4f}",
            f"- Bias (SurvivalEVAL − oracle) mean = {payload['bias_uno_survivaleval_mean']:+.4f}",
            f"- Δ bias = {payload['delta_bias_mean']:+.4f}",
            "",
            "Paper sentence:",
            (
                f"Under $\\tau = {args.tau:.2f}$, SurvivalEVAL reports Uno's C bias of "
                f"{payload['bias_uno_survivaleval_mean']:.4f} while scikit-survival reports "
                f"{payload['bias_uno_sksurv_mean']:.4f} "
                f"($\\Delta = {payload['delta_bias_mean']:+.4f}$)."
            ),
            "",
        ]
    )
    (out_dir / "ANCHOR_uno_backend_divergence.md").write_text(md)
    log(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
