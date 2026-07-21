"""
H04_paired_bootstrap_loo.py — Formal H4 close on LOO floor hits
==============================================================
Takes LOO ablations from ``F00_sens1_leave_one_out.json`` with ΔC ≥ 0.03,
refits each ablated Cox (same population as Fase A), and runs **paired
bootstrap** on ΔC = C_full − C_ablated (fixed risk scores; resample rows).

C00 / H4 decision: reject H0 if **any** hit has Δ ≥ 0.03 **and**
non-overlapping 95% CIs for C_full vs C_ablated (paired bootstrap on fixed risks).

Writes:
  results/probes/H04_paired_bootstrap_loo.{json,md}
  results/logs/H04_paired_bootstrap_loo.md

Execute:
    python -W default H04_paired_bootstrap_loo.py
    python -W default H04_paired_bootstrap_loo.py --full   # B=1000
"""

from __future__ import annotations

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
from src.metrics.io import utc_now, write_json

warnings.filterwarnings("default")

ALPHA = float(PROTOCOL["globals"]["alpha"])
DELTA_THR = float(PROTOCOL["hypotheses"]["H4"]["delta_c_threshold"])
SEED = int(PROTOCOL["globals"]["random_seed"])


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
        if (b + 1) % 200 == 0:
            log(f"        … bootstrap {b + 1}/{B}")
    lo_f, hi_f = np.quantile(boots_full, [0.025, 0.975])
    lo_a, hi_a = np.quantile(boots_ab, [0.025, 0.975])
    lo_d, hi_d = np.quantile(boots_delta, [0.025, 0.975])
    ci_nonoverlap = bool(lo_f > hi_a)
    return {
        "B": B,
        "boot_n": boot_n,
        "ci_full": [float(lo_f), float(hi_f)],
        "ci_ablated": [float(lo_a), float(hi_a)],
        "ci_delta": [float(lo_d), float(hi_d)],
        "ci_nonoverlap": ci_nonoverlap,
        "delta_mean_boot": float(boots_delta.mean()),
        "delta_se_boot": float(boots_delta.std(ddof=1)),
    }


def _to_md(payload: dict) -> str:
    lines = [
        "# H04 — Paired bootstrap on H4 LOO floor hits",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        f"Protocol: `{payload['protocol_version']}`",
        f"Δ threshold: **{DELTA_THR}**; B={payload['bootstrap_B']}; boot_n={payload['bootstrap_n']}",
        "",
        f"**H4 formal reject (∃ hit):** **{payload['h4_reject']}** "
        f"({payload['n_hits_reject']}/{payload['n_hits']} hits clear floor+CI)",
        "",
        "| Dropped | ΔC | CI_full | CI_ablated | nonoverlap | reject |",
        "|---------|-----|---------|------------|------------|--------|",
    ]
    for h in payload["hits"]:
        if h.get("status") != "live":
            lines.append(f"| {h.get('dropped')} | — | — | — | — | error |")
            continue
        cf = h["bootstrap"]["ci_full"]
        ca = h["bootstrap"]["ci_ablated"]
        lines.append(
            f"| `{h['dropped'][0]}` | {h['delta_c']:.4f} | "
            f"[{cf[0]:.4f},{cf[1]:.4f}] | [{ca[0]:.4f},{ca[1]:.4f}] | "
            f"{h['bootstrap']['ci_nonoverlap']} | **{h['reject_h4']}** |"
        )
    lines += ["", payload.get("note", ""), ""]
    return "\n".join(lines) + "\n"


def main() -> int:
    log("─" * 60)
    log("H04 — PAIRED BOOTSTRAP ON LOO H4 HITS")
    log("─" * 60)

    full_flag = "--full" in sys.argv
    B = 1000 if full_flag else min(200, int(cfg.EVAL["n_bootstrap"]))
    penalizer = float(cfg.DOMAIN_01.get("cox_penalizer", 0.01))

    sens_path = cfg.DIRS["probes"] / "F00_sens1_leave_one_out.json"
    if not sens_path.exists():
        log(f"ERROR: missing {sens_path} — run F00 sens1 first")
        return 1
    sens = json.loads(sens_path.read_text(encoding="utf-8"))
    loo_hits = [r for r in (sens.get("leave_one_out") or []) if r.get("ge_threshold")]
    log(f"  LOO hits: {len(loo_hits)}  B={B}  (--full={full_flag})")

    if not loo_hits:
        # Valid outcome after H6a lock: max LOO Δ can fall below threshold.
        payload = {
            "stage": "H04",
            "hypothesis": "H4",
            "generated_at_utc": utc_now(),
            "protocol_version": PROTOCOL["version"],
            "source_sens1": str(sens_path.relative_to(cfg.ROOT)),
            "decision_rule": PROTOCOL["hypotheses"]["H4"]["decision"],
            "delta_threshold": DELTA_THR,
            "bootstrap_B": B,
            "c_full": sens.get("c_full"),
            "max_delta_loo": sens.get("max_delta_loo"),
            "n_hits": 0,
            "n_hits_live": 0,
            "n_hits_reject": 0,
            "h4_reject": False,
            "hits": [],
            "note": (
                "No LOO ablations with ΔC ≥ threshold on current Cox population; "
                "H4 does not reject (no floor hits to bootstrap)."
            ),
            "cox_fit_population": cfg.DOMAIN_01.get("cox_fit_population"),
        }
        probes = cfg.DIRS["probes"]
        logs = cfg.DIRS["logs"]
        probes.mkdir(parents=True, exist_ok=True)
        logs.mkdir(parents=True, exist_ok=True)
        out_json = probes / "H04_paired_bootstrap_loo.json"
        out_md = probes / "H04_paired_bootstrap_loo.md"
        log_md = logs / "H04_paired_bootstrap_loo.md"
        write_json(out_json, payload)
        md = (
            f"# H04 paired bootstrap (LOO floor hits)\n\n"
            f"- LOO hits with Δ≥{DELTA_THR}: **0**\n"
            f"- C_full (from F00 sens1): `{payload.get('c_full')}`\n"
            f"- max Δ_LOO: `{payload.get('max_delta_loo')}`\n"
            f"- `h4_reject`: **False**\n\n"
            f"{payload['note']}\n"
        )
        out_md.write_text(md, encoding="utf-8")
        log_md.write_text(md, encoding="utf-8")
        log(f"Wrote {out_json}")
        log("H04 complete — no LOO hits; h4_reject=False")
        return 0

    full_path = cfg.DIRS["models"] / "domain1" / "cox_ahmed_green.joblib"
    full_blob = joblib.load(full_path)
    full_model: CoxPHFitter = full_blob["model"]
    full_cols = list(full_blob["smart_cols"])

    drives = cfg.DIRS["processed_d1"] / "drives.parquet"
    df = pd.read_parquet(drives)
    from src.domain1_cox_cohort import select_cox_fit_rows

    cohort = select_cox_fit_rows(df)
    use = cohort[["duration_days", "event"] + full_cols].dropna()
    use = use.loc[use["duration_days"] > 0].copy()
    time = use["duration_days"].to_numpy(float)
    event = use["event"].to_numpy(int)
    n = len(use)
    # Protocol cares about B=1000; boot_n=8k matches F00 default and stays tractable.
    boot_n = min(8_000, n)
    log(
        f"  n={n:,}  events={int(event.sum()):,}  boot_n={boot_n}  "
        f"pop={cfg.DOMAIN_01['cox_fit_population']}"
    )

    risk_full = np.asarray(
        full_model.predict_partial_hazard(use[full_cols]), float
    ).ravel()
    c_full = _harrell(time, event, risk_full)
    log(f"  C_full (frozen)={c_full:.4f}")

    hit_results = []
    for i, row in enumerate(sorted(loo_hits, key=lambda r: -float(r["delta_c"]))):
        dropped = list(row["dropped"])
        log(f"  [{i+1}/{len(loo_hits)}] drop {dropped} …")
        ablated_cols = [c for c in full_cols if c not in dropped]
        if len(ablated_cols) == len(full_cols):
            hit_results.append(
                {
                    "dropped": dropped,
                    "status": "error",
                    "error": "dropped cols not in full feature set",
                }
            )
            continue
        try:
            ablated = _fit_cox(use, ablated_cols, penalizer)
        except ConvergenceError as exc:
            hit_results.append(
                {"dropped": dropped, "status": "error", "error": str(exc)}
            )
            continue
        risk_ab = np.asarray(
            ablated.predict_partial_hazard(use[ablated_cols]), float
        ).ravel()
        c_ab = _harrell(time, event, risk_ab)
        delta_c = c_full - c_ab
        boot = _paired_bootstrap(
            time, event, risk_full, risk_ab, B=B, boot_n=boot_n, seed=SEED + i
        )
        delta_ge = bool(delta_c >= DELTA_THR)
        reject = bool(delta_ge and boot["ci_nonoverlap"])
        log(
            f"      C_ab={c_ab:.4f}  Δ={delta_c:.4f}  "
            f"CI_nonoverlap={boot['ci_nonoverlap']}  reject={reject}"
        )
        hit_results.append(
            {
                "dropped": dropped,
                "status": "live",
                "c_full": c_full,
                "c_ablated": c_ab,
                "delta_c": delta_c,
                "delta_threshold": DELTA_THR,
                "delta_ge_threshold": delta_ge,
                "sens1_delta_c": float(row.get("delta_c")),
                "bootstrap": boot,
                "reject_h4": reject,
            }
        )

    live = [h for h in hit_results if h.get("status") == "live"]
    n_rej = sum(1 for h in live if h.get("reject_h4"))
    h4_reject = n_rej > 0

    payload = {
        "stage": "H04",
        "hypothesis": "H4",
        "generated_at_utc": utc_now(),
        "protocol_version": PROTOCOL["version"],
        "source_sens1": str(sens_path.relative_to(cfg.ROOT)),
        "decision_rule": PROTOCOL["hypotheses"]["H4"]["decision"],
        "delta_threshold": DELTA_THR,
        "bootstrap_B": B,
        "bootstrap_n": boot_n,
        "full_n": n,
        "c_full": c_full,
        "n_hits": len(hit_results),
        "n_hits_live": len(live),
        "n_hits_reject": n_rej,
        "h4_reject": h4_reject,
        "hits": hit_results,
        "note": (
            "Paired bootstrap resamples rows with fixed full/ablated risk scores "
            "(same as F00). Formal H4 reject if any LOO floor hit has Δ≥threshold "
            "and non-overlapping CIs."
        ),
    }

    probes = cfg.DIRS["probes"]
    logs = cfg.DIRS["logs"]
    probes.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    out_json = probes / "H04_paired_bootstrap_loo.json"
    out_md = probes / "H04_paired_bootstrap_loo.md"
    log_md = logs / "H04_paired_bootstrap_loo.md"
    write_json(out_json, payload)
    md = _to_md(payload)
    out_md.write_text(md, encoding="utf-8")
    log_md.write_text(md, encoding="utf-8")
    log(f"Wrote {out_json}")
    log(f"H04 complete — h4_reject={h4_reject} ({n_rej}/{len(live)} hits)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
