"""F06 — Domain 1 cox_full CG-IPCW sensitivity sweep (artefatos.md).

Operational definition (docstring contract):
  Copula-Graphic replaces Kaplan–Meier in the censoring survival G(t) used
  for IPCW weights. Adjusted C = Uno's C with CG-based Ĝ; adjusted IBS =
  IPCW-IBS with the same substitution. Kendall τ = 0 ⇒ α → 0 ⇒ CG → KM.

Run:
  python results/paper/scripts/F06_copula_sweep_d1.py
  python results/paper/scripts/F06_copula_sweep_d1.py --smoke
  python results/paper/scripts/F06_copula_sweep_d1.py --B 200
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / "paper" / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results" / "paper" / "builders"))
sys.path.insert(0, str(ROOT / "results" / "paper" / "style"))

from numbers_io import load_numbers, v  # noqa: E402
from src.metrics.dependent_censoring import (  # noqa: E402
    PAPER_TAU_GRID,
    bootstrap_metric_sweep,
    sanity_tau0,
)
from src.metrics.freeze import load_frozen_manifest  # noqa: E402
from src.metrics.predict import predict_risk_scores, predict_survival_curves  # noqa: E402


OUT_JSON = ROOT / "results" / "paper" / "numbers_copula_sweep_d1.json"
OUT_FIG = ROOT / "results" / "paper" / "figures" / "F06_copula_sweep_d1.pdf"
JMLR_FIG = ROOT / "JMLR_submission" / "F06_copula_sweep_d1.pdf"


def _load_cox_full():
    manifest = load_frozen_manifest()
    art = next(
        a
        for a in manifest["artifacts"]
        if a.get("model_id") == "cox_full"
        and a.get("domain_id") == "DOMAIN_01"
        and a.get("present")
        and a.get("predict_ready")
    )
    risks = predict_risk_scores(art)
    curves = predict_survival_curves(art, n_grid=40)
    return art, risks, curves


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="3 τ × B=20 integration check")
    ap.add_argument("--B", type=int, default=200, help="Bootstrap replicates (default 200)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("F06 — loading frozen DOMAIN_01 cox_full …", flush=True)
    art, risks, curves = _load_cox_full()
    n = load_numbers()
    harrell_ref = float(v(n, "H1.DOMAIN_01.cox_full.C"))
    ibs_ref = float(v(n, "ladder.d01.cox_full.IBS"))

    print("F06 — τ=0 sanity (CG↔KM bridge + Harrell identity) …", flush=True)
    san = sanity_tau0(
        risks["risk"],
        risks["time"],
        risks["event"],
        curves["surv_grid"],
        curves["times_grid"],
        harrell_ref=harrell_ref,
        ibs_ref=ibs_ref,
    )
    print(
        f"  Harrell={san['harrell']:.6f} (ref {harrell_ref:.6f}); "
        f"Uno CG−KM={san['delta_uno_cg_minus_km']:.3g}; "
        f"IBS CG−KM={san['delta_ibs_cg_minus_km']:.3g}",
        flush=True,
    )

    if args.smoke:
        tau_grid = (0.0, 0.25, 0.75)
        B = 20
    else:
        tau_grid = PAPER_TAU_GRID
        B = int(args.B)

    print(f"F06 — sweep |τ|={len(tau_grid)} B={B} seed={args.seed} …", flush=True)
    t0 = time.time()
    sweep = bootstrap_metric_sweep(
        risks["risk"],
        risks["time"],
        risks["event"],
        curves["surv_grid"],
        curves["times_grid"],
        tau_grid=tau_grid,
        B=B,
        seed=int(args.seed),
        num_points=10,
    )
    elapsed = time.time() - t0
    print(f"  done in {elapsed / 60:.1f} min", flush=True)

    payload = {
        "model_id": "cox_full",
        "domain_id": "DOMAIN_01",
        "artifact_sha256": art.get("sha256"),
        "definition": sweep["definition"],
        "definition_short": (
            "Adjusted metrics = Uno C / IPCW-IBS with Copula-Graphic Ĝ(t) "
            "replacing Kaplan–Meier in IPCW censoring weights (Clayton α=2τ/(1−τ))."
        ),
        "tau_grid": sweep["tau_grid"],
        "clayton_alpha": sweep["clayton_alpha"],
        "uno_c_adjusted": sweep["uno_c_adjusted"],
        "ibs_adjusted": sweep["ibs_adjusted"],
        # aliases matching the original prompt schema
        "harrell_c_adjusted": sweep["uno_c_adjusted"],
        "reference": {
            "harrell_unadjusted": san["harrell"],
            "harrell_numbers_json": harrell_ref,
            "uno_km": san["uno_km"],
            "uno_cg_tau0": san["uno_cg_tau0"],
            "ibs_km": san["ibs_km"],
            "ibs_cg_tau0": san["ibs_cg_tau0"],
            "ibs_ladder_ref": ibs_ref,
            "note": (
                "Paper headline C=0.9595 is Harrell (no IPCW). The sweep metric is "
                "Uno C with CG-IPCW weights; at τ=0 it matches KM-Uno, not Harrell."
            ),
        },
        "sanity_tau0": san,
        "bootstrap": sweep["bootstrap"],
        "n": sweep["n"],
        "n_events": sweep["n_events"],
        "elapsed_sec": elapsed,
        "smoke": bool(args.smoke),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"  wrote {OUT_JSON}", flush=True)

    from figures import build_f06

    pdf = build_f06(payload)
    print(f"  wrote {pdf}", flush=True)
    # Optional local manuscript mirror (absent on the public reproduction surface)
    if not args.smoke and JMLR_FIG.parent.is_dir():
        shutil.copy2(pdf, JMLR_FIG)
        print(f"  copied {JMLR_FIG}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
