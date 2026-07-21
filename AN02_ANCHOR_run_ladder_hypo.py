"""
AN02_ANCHOR_run_ladder_hypo.py — Re-run anchor ladder hypothesis experiment
===========================================================================
Ports ``ladder_hypo.ipynb`` via ``src.anchor_harness``:

  CoxPH fixed; censoring ∈ {random, independent, Clayton τ};
  aggregate oracle CI/IBS → match paper Table 2 / Figure 5 errors.

Writes:
  results/ladder/anchor/seed_metrics.csv
  results/ladder/anchor/scenario_summary.json
  results/ladder/anchor/figure5_metric_errors.json
  results/ladder/anchor/an02_run_manifest.json

Execute:
    python -W default AN02_ANCHOR_run_ladder_hypo.py           # smoke (5 seeds)
    python -W default AN02_ANCHOR_run_ladder_hypo.py --full     # 100 seeds (paper)
    python -W default AN02_ANCHOR_run_ladder_hypo.py --seeds 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

# Avoid matplotlib font-cache races in sandbox / CI
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "mplconfig-anchor"))

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.anchor_harness import run_bias_experiment, summary_to_scenario_dict
from src.config import cfg
from src.metrics.io import write_json

warnings.filterwarnings("default")


def log(msg: str = "") -> None:
    print(msg, flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--full",
        action="store_true",
        help=f"Use paper n_seeds={cfg.ANCHOR['n_seeds_default']}",
    )
    p.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="Override number of seeds (default: smoke or --full)",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Override data_cfg n_samples (default: author 10000)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    log("─" * 60)
    log("AN02 — ANCHOR LADDER HYPO RUN")
    log("─" * 60)

    proto_path = cfg.ANCHOR["processed_dir"] / "anchor_dgp_protocol.json"
    if not proto_path.exists():
        from src.repro import waiting_return

        return waiting_return(f"{proto_path.relative_to(cfg.ROOT)} (run AN01).")

    if not (cfg.ANCHOR["raw_dir"] / "dgp.py").exists():
        from src.repro import waiting_return

        return waiting_return("author code (run AN00).")

    n_seeds = args.seeds
    if n_seeds is None:
        n_seeds = (
            cfg.ANCHOR["n_seeds_default"] if args.full else cfg.ANCHOR["n_seeds_smoke"]
        )
    mode = "full" if n_seeds >= cfg.ANCHOR["n_seeds_default"] else "smoke"

    data_cfg = dict(cfg.ANCHOR["data_cfg"])
    if args.n_samples is not None:
        data_cfg["n_samples"] = int(args.n_samples)

    out_dir = cfg.DIRS["ladder"] / "anchor"
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"  mode={mode}  seeds={n_seeds}  n_samples={data_cfg['n_samples']}")
    log("  Running bias experiment (author protocol)…")

    res, summary = run_bias_experiment(
        data_cfg=data_cfg,
        taus=(0.25, 0.5, 0.75),
        seeds=range(n_seeds),
    )

    seed_path = out_dir / "seed_metrics.csv"
    res.to_csv(seed_path, index=False)

    scenario = summary_to_scenario_dict(summary)
    summary_path = out_dir / "scenario_summary.json"
    write_json(
        summary_path,
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "n_seeds": n_seeds,
            "mode": mode,
            "data_cfg": data_cfg,
            "scenarios": scenario,
        },
    )

    # Flat dict expected by AN03 (scenario id → metrics)
    flat_path = out_dir / "scenario_summary_flat.json"
    write_json(flat_path, scenario)

    fig5 = {
        sid: {
            "bias_ci_harrell": v.get("bias_ci_harrell"),
            "bias_ci_uno": v.get("bias_ci_uno"),
            "bias_ipcw": v.get("bias_ipcw"),
        }
        for sid, v in scenario.items()
    }
    fig5_path = out_dir / "figure5_metric_errors.json"
    write_json(
        fig5_path,
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "n_seeds": n_seeds,
            "note": "Metric error = censored metric − oracle (Figure 5)",
            "errors": fig5,
        },
    )

    manifest = {
        "stage": "AN02",
        "role": "anchor_harness_check",
        "implementation_status": "live",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "n_seeds": n_seeds,
        "data_cfg": data_cfg,
        "outputs": {
            "seed_metrics": str(seed_path.relative_to(cfg.ROOT)),
            "scenario_summary": str(summary_path.relative_to(cfg.ROOT)),
            "scenario_summary_flat": str(flat_path.relative_to(cfg.ROOT)),
            "figure5_metric_errors": str(fig5_path.relative_to(cfg.ROOT)),
        },
        "table2_preview": {
            sid: {
                "n_events": round(v["n_events"], 1),
                "censor_pct": round(v["censor_pct"], 2),
                "ci_oracle": round(v["ci_oracle_mean"], 4),
                "ibs_oracle": round(v["ibs_oracle_mean"], 4),
            }
            for sid, v in scenario.items()
        },
    }
    man_path = out_dir / "an02_run_manifest.json"
    write_json(man_path, manifest)

    log(f"  Wrote {seed_path.relative_to(cfg.ROOT)}  ({len(res)} rows)")
    log(f"  Wrote {summary_path.relative_to(cfg.ROOT)}")
    log(f"  Wrote {fig5_path.relative_to(cfg.ROOT)}")
    log(f"  Wrote {man_path.relative_to(cfg.ROOT)}")
    for sid, prev in manifest["table2_preview"].items():
        log(
            f"    {sid:12s}  events={prev['n_events']:<8}  "
            f"cens%={prev['censor_pct']:<6}  "
            f"CI={prev['ci_oracle']}  IBS={prev['ibs_oracle']}"
        )
    log("AN02 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
