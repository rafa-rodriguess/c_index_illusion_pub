"""
E05_rung5_competing_risks.py — Ladder rung 5: Competing risks
=============================================================
Anchor double-helix ladder — rung 5 (competing risks).

Primary live exhibit: Bondora (DOMAIN_02 / H3) — naive default CIF vs
Aalen–Johansen CIF at 12 months (early repayment as competitor). No retrain.

Fine–Gray / cause-specific / Wolbers stay stub until separate ``*_cr`` fits.
DOMAIN_01 / DOMAIN_03 → not_applicable (no coded competitor in freeze).

Writes ``results/ladder/d0{n}_competing_risks.json``.

Execute:
    python -W default E05_rung5_competing_risks.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from src.config import cfg
from src.metrics.competing_risks import (
    build_cr_frame,
    cause_specific_eval,
    cif_bias,
    cif_bias_by_rating,
    fine_gray_eval,
    wolbers_concordance,
)
from src.metrics.freeze import load_frozen_manifest
from src.metrics.io import utc_now, write_json

warnings.filterwarnings("default")


def log(msg: str = "") -> None:
    print(msg, flush=True)


def _slug(domain_id: str) -> str:
    return domain_id.lower().replace("domain_", "d")


DOMAIN_CR_ROLE = {
    "DOMAIN_01": "not_applicable_no_competitor_coded",
    "DOMAIN_02": "primary_default_vs_prepayment",
    "DOMAIN_03": "not_applicable_no_competitor_coded",
}


def eval_domain_02() -> dict:
    path = cfg.ROOT / "data/processed/domain2/loans_split.parquet"
    if not path.exists():
        return {
            "implementation_status": "error",
            "error": f"missing {path}",
        }
    df = pd.read_parquet(path)
    # Primary: temporal test (same hold-out as Fase A / E01–E03)
    if "split_role" in df.columns:
        test = df.loc[df["split_role"] == "test"].copy()
        train = df.loc[df["split_role"] == "train"].copy()
    else:
        test = df
        train = df

    cr_test = build_cr_frame(test)
    # Overall bias on test (H3 statistic); B from EVAL (cap 1000)
    overall = cif_bias(
        cr_test["cr_time"].to_numpy(),
        cr_test["cr_event"].to_numpy(),
        n_bootstrap=min(int(cfg.EVAL["n_bootstrap"]), 1000),
        seed=int(getattr(cfg, "RANDOM_SEED", 42)),
    )
    # Strata: lighter bootstrap for speed
    strata = cif_bias_by_rating(
        test,
        n_bootstrap=200,
        seed=1,
    )

    # Train-side check (descriptive)
    cr_train = build_cr_frame(train)
    train_point = cif_bias(
        cr_train["cr_time"].to_numpy(),
        cr_train["cr_event"].to_numpy(),
        n_bootstrap=0,
        seed=0,
    )

    h3_met = bool(
        overall.get("supports_h3_direction")
        and strata.get("h3_strata_rule_met")
    )

    return {
        "implementation_status": "live",
        "eval_split": "test",
        "n_test": int(len(cr_test)),
        "n_train": int(len(cr_train)),
        "overall_test": overall,
        "by_rating_test": strata,
        "overall_train_point": {
            "delta": train_point.get("delta"),
            "naive": (train_point.get("naive") or {}).get("value"),
            "aj": (train_point.get("aj") or {}).get("value"),
        },
        "h3_preview": {
            "supports_direction_overall": overall.get("supports_h3_direction"),
            "strata_rule_met": strata.get("h3_strata_rule_met"),
            "h3_decision_preview": h3_met,
            "note": "Preview only — formal H3 after Holm in later stage.",
        },
        "cause_specific": cause_specific_eval(),
        "fine_gray": fine_gray_eval(),
        "wolbers": wolbers_concordance(),
    }


def main() -> int:
    log("─" * 60)
    log("E05 — RUNG 5 COMPETING RISKS (CIF naive vs AJ / H3)")
    log("─" * 60)

    manifest = load_frozen_manifest()
    domains = sorted({a["domain_id"] for a in manifest["artifacts"] if a["present"]})
    # Always include D2 if processed data exists even if only listed via manifest
    for d in ("DOMAIN_01", "DOMAIN_02", "DOMAIN_03"):
        if d not in domains:
            domains.append(d)
    domains = sorted(set(domains))

    out_paths = []
    for domain_id in domains:
        role = DOMAIN_CR_ROLE.get(domain_id, "unknown")
        if domain_id == "DOMAIN_02":
            body = eval_domain_02()
        else:
            body = {
                "implementation_status": "not_applicable",
                "note": role,
                "cause_specific": cause_specific_eval(),
                "fine_gray": fine_gray_eval(),
                "wolbers": wolbers_concordance(),
            }

        payload = {
            "stage": "E05",
            "rung": 5,
            "rung_name": "competing_risks",
            "anchor_parallel": "double-helix ladder rung 5 — competing risks",
            "generated_at_utc": utc_now(),
            "domain_id": domain_id,
            "cr_role": role,
            "hypothesis_hooks": ["H3"] if domain_id == "DOMAIN_02" else [],
            "rule": (
                "No overwrite of Fase A baselines. Nonparametric CIF only in this "
                "rung; *_cr model fits are separate."
            ),
            **body,
        }
        path = cfg.DIRS["ladder"] / f"{_slug(domain_id)}_competing_risks.json"
        write_json(path, payload)
        out_paths.append(path)

        st = payload.get("implementation_status")
        log(f"  {domain_id}: {st} ({role})")
        if domain_id == "DOMAIN_02" and st == "live":
            ov = payload["overall_test"]
            log(
                f"    Δ(naive−AJ)@12m={ov['delta']:.4f}  "
                f"CI=[{ov['bootstrap']['ci_low']:.4f},{ov['bootstrap']['ci_high']:.4f}]  "
                f"H3_dir={ov['supports_h3_direction']}"
            )
            stata = payload["by_rating_test"]
            log(
                f"    strata supporting H3: "
                f"{stata.get('n_strata_supporting_h3')}/{stata.get('n_strata')}  "
                f"rule_met={stata.get('h3_strata_rule_met')}"
            )

    log(f"E05 complete — {len(out_paths)} domain file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
