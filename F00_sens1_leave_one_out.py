"""
F00_sens1_leave_one_out.py — H4 primary survey (LOO + pilot subsets)
====================================================================
Leave-one-out SMART ablation on the frozen DOMAIN_01 Cox population, plus
all non-empty subsets of the pilot set ``cfg.H4_ABLATION_SMART`` ({5,197,198}).

Writes:
  results/probes/F00_sens1_leave_one_out.{json,md}

This is the **primary H4 exhibit** (existential survey). Formal close = H04
paired bootstrap on LOO hits with ΔC ≥ threshold.

Execute:
    python -W default F00_sens1_leave_one_out.py
"""

from __future__ import annotations

import itertools
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

THR = float(PROTOCOL["hypotheses"]["H4"]["delta_c_threshold"])


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


def _eval_drop(
    use: pd.DataFrame,
    full_cols: list[str],
    drop_cols: list[str],
    *,
    c_full: float,
    time: np.ndarray,
    event: np.ndarray,
    penalizer: float,
) -> dict:
    keep = [c for c in full_cols if c not in drop_cols]
    if not keep:
        return {
            "dropped": drop_cols,
            "c_ablated": None,
            "delta_c": None,
            "ge_threshold": False,
            "status": "error_empty_features",
        }
    try:
        model = _fit_cox(use, keep, penalizer)
        risk = np.asarray(model.predict_partial_hazard(use[keep]), float).ravel()
        c_ab = _harrell(time, event, risk)
        delta = float(c_full - c_ab)
        return {
            "dropped": list(drop_cols),
            "c_ablated": c_ab,
            "delta_c": delta,
            "ge_threshold": bool(delta >= THR),
            "status": "ok",
        }
    except ConvergenceError as exc:
        return {
            "dropped": list(drop_cols),
            "c_ablated": None,
            "delta_c": None,
            "ge_threshold": False,
            "status": f"convergence_error:{exc}",
        }


def main() -> int:
    log("─" * 60)
    log("F00_sens1 — LOO + H4 pilot subsets (PRIMARY H4 SURVEY)")
    log("─" * 60)

    full_path = cfg.DIRS["models"] / "domain1" / "cox_ahmed_green.joblib"
    if not full_path.exists():
        log("ERROR: missing full Cox — run D02_DOMAIN_01_train_cox.py")
        return 1

    blob = joblib.load(full_path)
    full_model: CoxPHFitter = blob["model"]
    full_cols = list(blob["smart_cols"])
    penalizer = float(cfg.DOMAIN_01.get("cox_penalizer", 0.01))

    drives = cfg.DIRS["processed_d1"] / "drives.parquet"
    df = pd.read_parquet(drives)
    from src.domain1_cox_cohort import select_cox_fit_rows

    cohort = select_cox_fit_rows(df)
    use = cohort[["duration_days", "event"] + full_cols].dropna()
    use = use.loc[use["duration_days"] > 0].copy()
    time = use["duration_days"].to_numpy(float)
    event = use["event"].to_numpy(int)

    risk_full = np.asarray(full_model.predict_partial_hazard(use[full_cols]), float).ravel()
    c_full = _harrell(time, event, risk_full)
    log(
        f"  n={len(use):,}  events={int(event.sum()):,}  C_full={c_full:.4f}  "
        f"thr={THR}  pop={cfg.DOMAIN_01['cox_fit_population']}"
    )

    # --- Leave-one-out ---
    loo: list[dict] = []
    for i, col in enumerate(full_cols, start=1):
        log(f"  LOO [{i}/{len(full_cols)}] drop {col} …")
        row = _eval_drop(
            use, full_cols, [col], c_full=c_full, time=time, event=event, penalizer=penalizer
        )
        loo.append(row)
        if row["status"] == "ok":
            mark = "HIT" if row["ge_threshold"] else "ok"
            log(f"    Δ={row['delta_c']:.4f}  C_ab={row['c_ablated']:.4f}  [{mark}]")
        else:
            log(f"    {row['status']}")

    # --- All non-empty subsets of pilot SMART set ---
    pilot_ids = list(cfg.H4_ABLATION_SMART)
    pilot_cols = [f"smart_{int(s)}_raw" for s in pilot_ids]
    missing = [c for c in pilot_cols if c not in full_cols]
    subsets: list[dict] = []
    if missing:
        log(f"  WARNING: pilot cols missing from full set: {missing}")
    else:
        n_sub = 2 ** len(pilot_cols) - 1
        log(f"  Pilot subsets of {pilot_ids} ({n_sub} non-empty) …")
        for r in range(1, len(pilot_cols) + 1):
            for combo in itertools.combinations(pilot_cols, r):
                ids = [int(c.replace("smart_", "").replace("_raw", "")) for c in combo]
                log(f"    subset {ids} …")
                row = _eval_drop(
                    use,
                    full_cols,
                    list(combo),
                    c_full=c_full,
                    time=time,
                    event=event,
                    penalizer=penalizer,
                )
                row["smart_ids"] = ids
                subsets.append(row)
                if row["status"] == "ok":
                    mark = "HIT" if row["ge_threshold"] else "ok"
                    log(f"      Δ={row['delta_c']:.4f}  [{mark}]")

    loo_hits = [r for r in loo if r.get("ge_threshold")]
    sub_hits = [r for r in subsets if r.get("ge_threshold")]
    max_loo = max((r["delta_c"] for r in loo if r.get("delta_c") is not None), default=None)
    max_sub = max((r["delta_c"] for r in subsets if r.get("delta_c") is not None), default=None)

    payload = {
        "stage": "F00_sensitivity",
        "test_id": 1,
        "test_name": "leave_one_out_and_h4_subsets_COMPLETE",
        "generated_at_utc": utc_now(),
        "protocol_version": PROTOCOL["version"],
        "c_full": c_full,
        "delta_threshold": THR,
        "leave_one_out": loo,
        "h4_subsets": subsets,
        "pilot_smart_ids": pilot_ids,
        "loo_hits_ge_threshold": len(loo_hits),
        "h4_subset_hits_ge_threshold": len(sub_hits),
        "max_delta_loo": max_loo,
        "max_delta_h4_subset": max_sub,
        "note": (
            "Primary H4 survey (existential). Formal close via H04 bootstrap "
            "on LOO hits with Δ>=threshold."
        ),
    }

    out_dir = cfg.DIRS["probes"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "F00_sens1_leave_one_out.json"
    out_md = out_dir / "F00_sens1_leave_one_out.md"
    write_json(out_json, payload)

    lines = [
        "# F00_sens1 — Leave-one-out + pilot subsets (PRIMARY H4)",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        f"C_full={c_full:.4f}  threshold={THR}",
        f"LOO hits ≥ thr: **{len(loo_hits)}** / {len(loo)}  (max Δ={max_loo})",
        f"Pilot-subset hits ≥ thr: **{len(sub_hits)}** / {len(subsets)}  (max Δ={max_sub})",
        "",
        "## LOO hits",
        "",
        "| Dropped | C_ablated | ΔC |",
        "|---------|-----------|-----|",
    ]
    for r in sorted(loo_hits, key=lambda x: -(x.get("delta_c") or 0)):
        lines.append(
            f"| `{','.join(r['dropped'])}` | {r['c_ablated']:.4f} | {r['delta_c']:.4f} |"
        )
    lines += [
        "",
        "## Pilot subsets hits",
        "",
        "| SMART ids | ΔC |",
        "|-----------|-----|",
    ]
    for r in sorted(sub_hits, key=lambda x: -(x.get("delta_c") or 0)):
        lines.append(f"| {r.get('smart_ids')} | {r['delta_c']:.4f} |")
    lines.append("")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"Wrote {out_json}")
    log(f"Wrote {out_md}")
    log(f"F00_sens1 complete — LOO hits={len(loo_hits)}  subset hits={len(sub_hits)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
