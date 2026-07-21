"""
F00b_h4_cluster_ablation.py — H4 extension: theoretically motivated SMART clusters
=================================================================================
Paired-bootstrap ablation on Domain 1 ``cox_full`` (H6a cohort), removing
feature *clusters* simultaneously (not LOO).

Clusters (SMART IDs → ``smart_{id}_raw``):
  A — age/usage proxy: {9, 240, 241}
  B — degradation trio: {5, 197, 198}  (= cfg.H4_ABLATION_SMART)
  C — combined A∪B: {5, 9, 197, 198, 240, 241}

H4 decision (same as freeze): reject exists if ΔC ≥ 0.03 AND non-overlapping
95% CIs (paired bootstrap on fixed risks).

Writes:
  results/probes/h4_cluster_ablation.json
  results/probes/h4_cluster_ablation.md

Execute:
    python -W default F00b_h4_cluster_ablation.py
    python -W default F00b_h4_cluster_ablation.py --full   # B=1000
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
from sksurv.metrics import concordance_index_censored

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import PROTOCOL, cfg
from src.domain1_cox_cohort import select_cox_fit_rows
from src.metrics.io import utc_now, write_json
from src.repro import add_strict_arg, waiting_return

warnings.filterwarnings("default")

DELTA_THR = float(PROTOCOL["hypotheses"]["H4"]["delta_c_threshold"])
SEED = int(PROTOCOL["globals"]["random_seed"])

CLUSTERS: dict[str, list[int]] = {
    "A_age_usage": [9, 240, 241],
    "B_degradation": [5, 197, 198],
    "C_combined_AB": [5, 9, 197, 198, 240, 241],
}


def log(msg: str = "") -> None:
    print(msg, flush=True)


def _harrell(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    return float(
        concordance_index_censored(
            event.astype(bool), time.astype(float), risk.astype(float)
        )[0]
    )


def _fit_cox(data: pd.DataFrame, fit_cols: list[str], penalizer: float) -> CoxPHFitter:
    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(
        data[["duration_days", "event"] + fit_cols],
        duration_col="duration_days",
        event_col="event",
        show_progress=False,
    )
    return cph


def _paired_bootstrap(
    time: np.ndarray,
    event: np.ndarray,
    risk_full: np.ndarray,
    risk_ab: np.ndarray,
    *,
    B: int,
    boot_n: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    n = len(time)
    boots_full = np.empty(B)
    boots_ab = np.empty(B)
    boots_delta = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=boot_n)
        bf = _harrell(time[idx], event[idx], risk_full[idx])
        ba = _harrell(time[idx], event[idx], risk_ab[idx])
        boots_full[b] = bf
        boots_ab[b] = ba
        boots_delta[b] = bf - ba
        if (b + 1) % 250 == 0:
            log(f"        … bootstrap {b + 1}/{B}")
    lo_f, hi_f = np.quantile(boots_full, [0.025, 0.975])
    lo_a, hi_a = np.quantile(boots_ab, [0.025, 0.975])
    lo_d, hi_d = np.quantile(boots_delta, [0.025, 0.975])
    return {
        "B": B,
        "boot_n": boot_n,
        "ci_full": [float(lo_f), float(hi_f)],
        "ci_ablated": [float(lo_a), float(hi_a)],
        "ci_delta": [float(lo_d), float(hi_d)],
        "ci_nonoverlap": bool(lo_f > hi_a),
        "delta_mean_boot": float(boots_delta.mean()),
        "delta_se_boot": float(boots_delta.std(ddof=1)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use B=1000 (protocol). Default: min(200, n_bootstrap).",
    )
    add_strict_arg(parser)
    args = parser.parse_args()

    B = (
        int(cfg.EVAL["n_bootstrap"])
        if args.full
        else min(200, int(cfg.EVAL["n_bootstrap"]))
    )

    model_path = cfg.DIRS["models_d1"] / "cox_ahmed_green.joblib"
    parquet = cfg.DIRS["processed_d1"] / "drives.parquet"
    if not model_path.exists() or not parquet.exists():
        return waiting_return(
            "missing Domain 1 model or drives.parquet — run D00–D02 first",
            strict=args.strict,
        )

    artifact = joblib.load(model_path)
    full_model: CoxPHFitter = artifact["model"]
    full_cols: list[str] = list(artifact["smart_cols"])
    penalizer = float(artifact.get("penalizer", cfg.DOMAIN_01.get("cox_penalizer", 0.01)))

    df = pd.read_parquet(parquet)
    cohort = select_cox_fit_rows(df)
    use = cohort[["duration_days", "event"] + full_cols].dropna()
    use = use[use["duration_days"] > 0].reset_index(drop=True)
    time = use["duration_days"].to_numpy(float)
    event = use["event"].to_numpy(int)
    boot_n = min(8000, len(use))

    risk_full = np.asarray(
        full_model.predict_partial_hazard(use[full_cols]), float
    ).ravel()
    c_full = _harrell(time, event, risk_full)
    log(f"F00b: n={len(use)}  C_full={c_full:.6f}  B={B}  boot_n={boot_n}")

    results = []
    any_reject = False
    for i, (cid, smart_ids) in enumerate(CLUSTERS.items()):
        drop_cols = [f"smart_{s}_raw" for s in smart_ids]
        missing = [c for c in drop_cols if c not in full_cols]
        keep = [c for c in full_cols if c not in drop_cols]
        log(f"  cluster {cid}: drop {smart_ids}")
        if missing:
            results.append(
                {
                    "cluster_id": cid,
                    "smart_ids": smart_ids,
                    "dropped": drop_cols,
                    "status": "error_missing_features",
                    "missing": missing,
                }
            )
            continue
        try:
            cph_ab = _fit_cox(use, keep, penalizer)
            risk_ab = np.asarray(
                cph_ab.predict_partial_hazard(use[keep]), float
            ).ravel()
            c_ab = _harrell(time, event, risk_ab)
            delta = float(c_full - c_ab)
            boot = _paired_bootstrap(
                time,
                event,
                risk_full,
                risk_ab,
                B=B,
                boot_n=boot_n,
                seed=SEED + 100 + i,
            )
            ge = bool(delta >= DELTA_THR)
            reject = bool(ge and boot["ci_nonoverlap"])
            any_reject = any_reject or reject
            results.append(
                {
                    "cluster_id": cid,
                    "smart_ids": smart_ids,
                    "dropped": drop_cols,
                    "c_ablated": c_ab,
                    "delta_c": delta,
                    "ge_threshold": ge,
                    "bootstrap": boot,
                    "reject_h4_cluster": reject,
                    "status": "ok",
                }
            )
            log(
                f"    C_ab={c_ab:.6f}  ΔC={delta:+.6f}  "
                f"nonoverlap={boot['ci_nonoverlap']}  reject={reject}"
            )
        except ConvergenceError as exc:
            results.append(
                {
                    "cluster_id": cid,
                    "smart_ids": smart_ids,
                    "dropped": drop_cols,
                    "status": "error_convergence",
                    "error": str(exc),
                }
            )

    payload = {
        "stage": "F00b",
        "hypothesis": "H4_cluster_extension",
        "protocol_version": PROTOCOL["version"],
        "decision_rule": (
            "Reject exists if ΔC >= 0.03 AND non-overlapping full vs ablated CIs "
            "(paired bootstrap; same rule as H4 freeze)."
        ),
        "delta_threshold": DELTA_THR,
        "bootstrap_B": B,
        "bootstrap_n": boot_n,
        "random_seed": SEED,
        "c_full": c_full,
        "cox_fit_population": cfg.DOMAIN_01["cox_fit_population"],
        "n": int(len(use)),
        "clusters": results,
        "any_cluster_reject": any_reject,
        "generated_at_utc": utc_now(),
    }

    out_json = cfg.DIRS["probes"] / "h4_cluster_ablation.json"
    write_json(out_json, payload)

    lines = [
        "# H4 cluster ablation (Domain 1)",
        "",
        f"C_full = **{c_full:.6f}**; Δ threshold = {DELTA_THR}; B={B}",
        f"**Any cluster rejects H4 rule:** **{any_reject}**",
        "",
        "| Cluster | SMART IDs | C_ablated | ΔC | CI_full | CI_ablated | nonoverlap | reject |",
        "|---|---|---:|---:|---|---|---|---|",
    ]
    for r in results:
        if r.get("status") != "ok":
            lines.append(
                f"| {r['cluster_id']} | {r['smart_ids']} | — | — | — | — | — | {r['status']} |"
            )
            continue
        b = r["bootstrap"]
        lines.append(
            f"| {r['cluster_id']} | {r['smart_ids']} | {r['c_ablated']:.6f} | "
            f"{r['delta_c']:+.6f} | "
            f"[{b['ci_full'][0]:.4f},{b['ci_full'][1]:.4f}] | "
            f"[{b['ci_ablated'][0]:.4f},{b['ci_ablated'][1]:.4f}] | "
            f"{b['ci_nonoverlap']} | {r['reject_h4_cluster']} |"
        )
    md_path = cfg.DIRS["probes"] / "h4_cluster_ablation.md"
    md_path.write_text("\n".join(lines) + "\n")
    log(f"Wrote {out_json.relative_to(cfg.ROOT)}")
    log(f"Wrote {md_path.relative_to(cfg.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
