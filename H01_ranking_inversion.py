"""
H01_ranking_inversion.py — Formal H1: C-index vs IPCW-IBS ranking (τ_K)
=======================================================================
C00.1 (v5 amendment) / C00.2:

  Primary: subject-level stratified bootstrap of τ_K (B=1000 with ``--full``).
  Reject domain H0 iff τ_obs ≤ 0.5 AND bootstrap CI upper ≤ 0.5.

  Sensitivity: rank-permutation of model labels (unreachable p for k≤3;
  retained for honesty / appendix).

IBS backend is **sksurv for all models** (fair within-domain).
Harrell C / IBS are recomputed on bootstrap draws of the eval set from
cached risk scores and S(t|x) (no retrain).

Writes:
  results/probes/H01_ranking_inversion.{json,md}
  results/logs/H01_ranking_inversion.md

Execute:
    python -W default H01_ranking_inversion.py          # smoke B=200
    python -W default H01_ranking_inversion.py --full   # B=1000
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau
from sksurv.metrics import concordance_index_censored

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import PROTOCOL, cfg
from src.metrics.freeze import load_frozen_manifest
from src.metrics.io import utc_now, write_json
from src.metrics.predict import load_train_for_ipcw, predict_survival_curves
from src.metrics.proper_scores import ipcw_integrated_brier_sksurv

warnings.filterwarnings("default")

N_GRID = 40
ALPHA = float(PROTOCOL["globals"]["alpha"])
N_PERM = int(PROTOCOL["globals"]["n_permutation"])
SEED = int(PROTOCOL["globals"]["random_seed"])
B_FULL = int(PROTOCOL["globals"]["n_bootstrap"])
B_SMOKE = 50
BOOT_N_CAP = 5000  # stratified subsample per bootstrap when eval n is larger

RANKINGS: dict[str, list[str]] = {
    "DOMAIN_01": list(PROTOCOL["freeze_decisions"]["C00.2"]["rankings"]["domain1"]),
    "DOMAIN_02": list(PROTOCOL["freeze_decisions"]["C00.2"]["rankings"]["domain2"]),
    "DOMAIN_03": list(PROTOCOL["freeze_decisions"]["C00.2"]["rankings"]["domain3"]),
}


def log(msg: str = "") -> None:
    print(msg, flush=True)


def _slug(domain_id: str) -> str:
    return domain_id.lower().replace("domain_", "d")


def _harrell(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    return float(
        concordance_index_censored(
            event.astype(bool), time.astype(float), risk.astype(float)
        )[0]
    )


def _rank_high_better(scores: dict[str, float], keys: list[str]) -> list[str]:
    return sorted(keys, key=lambda k: (-scores[k], k))


def _rank_low_better(scores: dict[str, float], keys: list[str]) -> list[str]:
    return sorted(keys, key=lambda k: (scores[k], k))


def _order_to_ranks(order: list[str], keys: list[str]) -> np.ndarray:
    pos = {m: i for i, m in enumerate(order)}
    return np.asarray([pos[k] for k in keys], dtype=float)


def kendall_tau_orders(order_c: list[str], order_ibs: list[str], keys: list[str]) -> float:
    r1 = _order_to_ranks(order_c, keys)
    r2 = _order_to_ranks(order_ibs, keys)
    tau, _ = kendalltau(r1, r2)
    return float(tau)


def permutation_pvalue_tau(
    order_c: list[str],
    order_ibs: list[str],
    keys: list[str],
    *,
    tau_obs: float,
    n_perm: int = N_PERM,
    seed: int = SEED,
) -> dict:
    """Sensitivity-only: permute IBS ranking labels (unreachable for k≤3)."""
    k = len(keys)
    n_exact = int(math.factorial(k)) if k <= 12 else None
    rng = np.random.default_rng(seed)

    def _tau_for_ibs_order(ibs_ord: list[str]) -> float:
        return kendall_tau_orders(order_c, ibs_ord, keys)

    if n_exact is not None and n_exact <= n_perm:
        taus = [_tau_for_ibs_order(list(perm)) for perm in itertools.permutations(keys)]
        method = "exact_enumeration"
        n_used = n_exact
    else:
        taus = []
        for _ in range(n_perm):
            perm = list(keys)
            rng.shuffle(perm)
            taus.append(_tau_for_ibs_order(perm))
        method = "monte_carlo"
        n_used = n_perm

    arr = np.asarray(taus, dtype=float)
    p = float((np.sum(arr <= tau_obs) + 1.0) / (len(arr) + 1.0))
    return {
        "method": method,
        "role": "sensitivity_only",
        "n_perm_requested": n_perm,
        "n_perm_used": n_used,
        "n_models": k,
        "min_attainable_p_unsmoothed": 1.0 / n_used if n_used else None,
        "p_value": p,
        "tau_null_mean": float(arr.mean()),
        "tau_null_min": float(arr.min()),
        "tau_obs": tau_obs,
    }


def _cache_model(artifact: dict, times_grid: np.ndarray | None) -> dict:
    curves = predict_survival_curves(artifact, n_grid=N_GRID, times_grid=times_grid)
    train = load_train_for_ipcw(artifact)
    if train is None:
        raise RuntimeError(f"IPCW train missing for {artifact['model_id']}")
    return {
        "model_id": artifact["model_id"],
        "time": np.asarray(curves["time"], dtype=float),
        "event": np.asarray(curves["event"], dtype=int),
        "risk": np.asarray(curves["risk"], dtype=float),
        "surv_grid": np.asarray(curves["surv_grid"], dtype=float),
        "times_grid": np.asarray(curves["times_grid"], dtype=float),
        "curve_backend": curves["curve_backend"],
        "train_time": np.asarray(train[0], dtype=float),
        "train_event": np.asarray(train[1], dtype=int),
        "n": int(curves["n"]),
    }


def _metrics_on_idx(cache: dict, idx: np.ndarray) -> tuple[float, float]:
    t = cache["time"][idx]
    e = cache["event"][idx]
    r = cache["risk"][idx]
    S = cache["surv_grid"][:, idx]
    c = _harrell(t, e, r)
    ibs = ipcw_integrated_brier_sksurv(
        time_train=cache["train_time"],
        event_train=cache["train_event"],
        time_test=t,
        event_test=e,
        surv_grid=S,
        times_grid=cache["times_grid"],
    )
    if ibs.get("status") != "live" or ibs.get("value") is None:
        raise RuntimeError(f"sksurv IBS failed for {cache['model_id']}: {ibs}")
    return c, float(ibs["value"])


def _stratified_indices(
    event: np.ndarray,
    rng: np.random.Generator,
    *,
    boot_n: int | None = None,
) -> np.ndarray:
    event = np.asarray(event, dtype=int)
    n = len(event)
    pos = np.where(event == 1)[0]
    neg = np.where(event == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        size = n if boot_n is None else min(n, boot_n)
        return rng.integers(0, n, size=size)

    if boot_n is None or boot_n >= n:
        n_pos, n_neg = len(pos), len(neg)
    else:
        # Keep event rate; allocate boot_n across strata
        rate = len(pos) / n
        n_pos = max(1, int(round(boot_n * rate)))
        n_neg = max(1, boot_n - n_pos)
        if n_pos + n_neg > boot_n:
            n_neg = boot_n - n_pos

    boot_pos = rng.choice(pos, size=n_pos, replace=True)
    boot_neg = rng.choice(neg, size=n_neg, replace=True)
    return np.concatenate([boot_pos, boot_neg])


def subject_bootstrap_tau(
    caches: dict[str, dict],
    keys: list[str],
    *,
    B: int,
    seed: int,
    alpha: float = ALPHA,
    boot_n: int | None = BOOT_N_CAP,
) -> dict:
    """
    Stratified subject bootstrap of τ_K between C-ranking and IBS-ranking.

    Predictions are fixed; only eval-row indices are resampled (paired across
    models — same indices for every model in the domain).
    """
    ref = caches[keys[0]]
    n = ref["n"]
    for k in keys[1:]:
        if caches[k]["n"] != n:
            raise RuntimeError(
                f"Eval n mismatch within domain: {keys[0]}={n} vs {k}={caches[k]['n']}"
            )
        if not np.allclose(caches[k]["time"], ref["time"]):
            raise RuntimeError(f"time vector mismatch: {k}")
        if not np.array_equal(caches[k]["event"], ref["event"]):
            raise RuntimeError(f"event vector mismatch: {k}")

    effective_boot_n = None if boot_n is None or boot_n >= n else int(boot_n)
    rng = np.random.default_rng(seed)
    taus = np.empty(B, dtype=float)
    inversions = np.empty(B, dtype=bool)

    for b in range(B):
        if B >= 50 and (b + 1) % max(1, B // 10) == 0:
            log(f"      bootstrap {b + 1}/{B} …")
        idx = _stratified_indices(ref["event"], rng, boot_n=effective_boot_n)
        harrell: dict[str, float] = {}
        ibs_scores: dict[str, float] = {}
        for mid in keys:
            c, ibs = _metrics_on_idx(caches[mid], idx)
            harrell[mid] = c
            ibs_scores[mid] = ibs
        order_c = _rank_high_better(harrell, keys)
        order_ibs = _rank_low_better(ibs_scores, keys)
        taus[b] = kendall_tau_orders(order_c, order_ibs, keys)
        inversions[b] = order_c != order_ibs

    lo = float(np.quantile(taus, alpha / 2.0))
    hi = float(np.quantile(taus, 1.0 - alpha / 2.0))
    p_tau_eq_1 = float((np.sum(taus >= 1.0 - 1e-12) + 1.0) / (B + 1.0))
    return {
        "method": "subject_level_stratified_bootstrap",
        "role": "primary_C00_1",
        "B": B,
        "n_eval": n,
        "boot_n": effective_boot_n or n,
        "alpha": alpha,
        "ci": [lo, hi],
        "ci_upper": hi,
        "ci_lower": lo,
        "tau_boot_mean": float(taus.mean()),
        "tau_boot_min": float(taus.min()),
        "tau_boot_max": float(taus.max()),
        "inversion_rate": float(inversions.mean()),
        "p_tau_eq_1": p_tau_eq_1,
        "reject_gate_ci_upper_le_0_5": bool(hi <= 0.5),
    }


def eval_domain(domain_id: str, artifacts_by_id: dict[str, dict], *, B: int) -> dict:
    keys = list(RANKINGS[domain_id])
    eval_order = sorted(keys, key=lambda m: 0 if "xgb" in m else 1)

    caches: dict[str, dict] = {}
    times_grid = None
    for mid in eval_order:
        art = artifacts_by_id.get(mid)
        if art is None:
            return {
                "domain_id": domain_id,
                "status": "error",
                "error": f"Artifact missing for {mid}",
                "ranking_keys": keys,
            }
        caches[mid] = _cache_model(art, times_grid)
        if times_grid is None:
            times_grid = caches[mid]["times_grid"]

    # Observed metrics on full eval set
    harrell: dict[str, float] = {}
    ibs_scores: dict[str, float] = {}
    ibs_meta: dict[str, dict] = {}
    full_idx = np.arange(caches[keys[0]]["n"])
    for mid in keys:
        c, ibs = _metrics_on_idx(caches[mid], full_idx)
        harrell[mid] = c
        ibs_scores[mid] = ibs
        ibs_meta[mid] = {
            "value": ibs,
            "curve_backend": caches[mid]["curve_backend"],
            "n": caches[mid]["n"],
            "n_grid": N_GRID,
            "metric_backend": "sksurv",
        }

    order_c = _rank_high_better(harrell, keys)
    order_ibs = _rank_low_better(ibs_scores, keys)
    tau = kendall_tau_orders(order_c, order_ibs, keys)
    perm = permutation_pvalue_tau(order_c, order_ibs, keys, tau_obs=tau)
    boot = subject_bootstrap_tau(caches, keys, B=B, seed=SEED + hash(domain_id) % 10_000)

    reject = bool(tau <= 0.5 and boot["p_tau_eq_1"] < ALPHA)
    p_family = float(boot["p_tau_eq_1"])

    return {
        "domain_id": domain_id,
        "status": "live",
        "ranking_keys": keys,
        "harrell_c": {k: harrell[k] for k in keys},
        "ipcw_ibs_sksurv": {k: ibs_scores[k] for k in keys},
        "ibs_meta": ibs_meta,
        "order_by_c_desc": order_c,
        "order_by_ibs_asc": order_ibs,
        "binary_inversion": order_c != order_ibs,
        "tau_K": tau,
        "bootstrap": boot,
        "permutation_sensitivity": perm,
        "alpha": ALPHA,
        "reject_H0_C00_1": reject,
        "p_family": p_family,
        "decision_note": (
            f"C00.1 v5: tau_K={tau:.4f} <= 0.5 → {tau <= 0.5}; "
            f"boot p(τ=1)={boot['p_tau_eq_1']:.4g} < {ALPHA} → {boot['p_tau_eq_1'] < ALPHA}; "
            f"reject={reject}; CI={boot['ci']}"
        ),
    }


def _to_md(payload: dict) -> str:
    lines = [
        "# H01 — Ranking inversion (C vs IPCW-IBS)",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        f"Protocol: `{payload['protocol_version']}`",
        f"IBS backend: **sksurv** (n_grid={N_GRID}) for all C00.2 models.",
        f"Primary test: subject-level bootstrap B={payload['B']} (C00.1 v5).",
        "",
        "| Domain | τ_K | inversion | boot CI | reject C00.1 | p_family | C order | IBS order |",
        "|--------|-----|-----------|---------|--------------|----------|---------|-----------|",
    ]
    for d in payload["domains"]:
        if d.get("status") != "live":
            lines.append(
                f"| {d['domain_id']} | — | — | — | error | — | {d.get('error', '')} | — |"
            )
            continue
        ci = d["bootstrap"]["ci"]
        lines.append(
            f"| {d['domain_id']} | {d['tau_K']:.3f} | {d['binary_inversion']} | "
            f"[{ci[0]:.3f}, {ci[1]:.3f}] | **{d['reject_H0_C00_1']}** | "
            f"{d['p_family']:.4g} | "
            f"`{' > '.join(d['order_by_c_desc'])}` | "
            f"`{' < '.join(d['order_by_ibs_asc'])}` |"
        )
    lines += [
        "",
        f"**Inner Holm (min domain p_family):** {payload['inner_holm']['min_p']:.6g}",
        "",
        f"**H1 domain-level rejects:** {payload['n_domain_rejects']} / {payload['n_domains_live']}",
        "",
        "## Amendment note",
        "",
        payload["amendment_note"],
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--full",
        action="store_true",
        help=f"B={B_FULL} (protocol); default smoke B={B_SMOKE}; boot_n cap={BOOT_N_CAP}",
    )
    args = ap.parse_args()
    B = B_FULL if args.full else B_SMOKE

    log("─" * 60)
    log("H01 — RANKING INVERSION (τ_K + subject bootstrap C00.1 v5)")
    log(f"B={B}  boot_n_cap={BOOT_N_CAP}  ({'full' if args.full else 'smoke'})")
    log("─" * 60)

    manifest = load_frozen_manifest()
    by_id = {
        a["model_id"]: a
        for a in manifest["artifacts"]
        if a.get("present") and a.get("kind") == "estimator"
    }

    domain_order = ["DOMAIN_02", "DOMAIN_01", "DOMAIN_03"]
    results = []
    for domain_id in domain_order:
        log(f"  evaluating {domain_id} …")
        r = eval_domain(domain_id, by_id, B=B)
        results.append(r)
        if r.get("status") == "live":
            log(
                f"    τ_K={r['tau_K']:.3f}  inversion={r['binary_inversion']}  "
                f"CI=[{r['bootstrap']['ci'][0]:.3f},{r['bootstrap']['ci'][1]:.3f}]  "
                f"reject={r['reject_H0_C00_1']}  p_family={r['p_family']:.4g}"
            )
            log(f"    C:   {' > '.join(r['order_by_c_desc'])}")
            ibs_bits = ", ".join(
                f"{k}={r['ipcw_ibs_sksurv'][k]:.4f}" for k in r["ranking_keys"]
            )
            log(f"    IBS: {' < '.join(r['order_by_ibs_asc'])}  ({ibs_bits})")
        else:
            log(f"    ERROR {r.get('error')}")

    order_idx = {"DOMAIN_01": 0, "DOMAIN_02": 1, "DOMAIN_03": 2}
    results.sort(key=lambda d: order_idx.get(d["domain_id"], 9))

    live = [d for d in results if d.get("status") == "live"]
    ps = [d["p_family"] for d in live]
    min_p = float(min(ps)) if ps else None
    n_reject = sum(1 for d in live if d.get("reject_H0_C00_1"))

    amend = (
        "PROTOCOL 2026-07-12.c00.v5 amends C00.1: formal gate is subject-level "
        "stratified bootstrap p(τ_K=1) < α together with τ_obs ≤ 0.5. "
        "Percentile CI is reported. Rank-permutation remains sensitivity-only "
        "because k≤3 makes p < α unreachable (min unsmoothed p = 1/k!)."
    )

    payload = {
        "stage": "H01",
        "hypothesis": "H1",
        "generated_at_utc": utc_now(),
        "protocol_version": PROTOCOL["version"],
        "decision_rule": PROTOCOL["freeze_decisions"]["C00.1"]["official"],
        "ibs_backend": "sksurv.metrics.integrated_brier_score",
        "n_grid": N_GRID,
        "B": B,
        "n_permutation_sensitivity": N_PERM,
        "alpha": ALPHA,
        "domains": results,
        "n_domains_live": len(live),
        "n_domain_rejects": n_reject,
        "inner_holm": {
            "rule": "min of domain p_family (formal outer Holm in G03)",
            "min_p": min_p,
            "domain_ps": {d["domain_id"]: d["p_family"] for d in live},
        },
        # back-compat aliases for G03 / older readers
        "inner_holm_preview": {
            "rule": "min of domain p_family (formal Holm in G03)",
            "min_p": min_p,
            "domain_ps": {d["domain_id"]: d["p_family"] for d in live},
        },
        "amendment_note": amend,
        "any_binary_inversion": any(d.get("binary_inversion") for d in live),
        "formal_reject_C00_1_any_domain": n_reject > 0,
    }

    out_json = cfg.DIRS["probes"] / "H01_ranking_inversion.json"
    out_md = cfg.DIRS["probes"] / "H01_ranking_inversion.md"
    log_md = cfg.DIRS["logs"] / "H01_ranking_inversion.md"
    cfg.DIRS["probes"].mkdir(parents=True, exist_ok=True)
    cfg.DIRS["logs"].mkdir(parents=True, exist_ok=True)
    write_json(out_json, payload)
    md = _to_md(payload)
    out_md.write_text(md, encoding="utf-8")
    log_md.write_text(md, encoding="utf-8")
    log(f"Wrote {out_json}")
    log(f"Wrote {out_md}")
    log(
        f"H01 complete — domain rejects={n_reject}/{len(live)}; "
        f"any inversion={payload['any_binary_inversion']}; "
        f"formal C00.1 any={payload['formal_reject_C00_1_any_domain']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
