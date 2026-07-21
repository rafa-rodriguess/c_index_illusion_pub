"""
AN03_ANCHOR_gap.py — Paper-ready harness gap vs Lillelund Table 2
=================================================================
Builds ``results/reproduction/ANCHOR_reproduction_table.{json,csv,md,tex}``
with the same schema as Domain lanes (``src.reproduction_table``).

Primary targets: Table 2 oracle CI / IBS (and event counts) per censoring
scenario. Figure 5 metric-error curves are auxiliary until AN02 is live.

Execute:
    python -W default AN03_ANCHOR_gap.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import cfg
from src.reproduction_table import build_document, row, write_reproduction_table

warnings.filterwarnings("default")


def log(msg: str = "") -> None:
    print(msg, flush=True)


def _load_ours_summary() -> dict | None:
    """Prefer flat scenario dict written by AN02; fall back to nested."""
    flat = cfg.DIRS["ladder"] / "anchor" / "scenario_summary_flat.json"
    nested = cfg.DIRS["ladder"] / "anchor" / "scenario_summary.json"
    if flat.exists():
        return json.loads(flat.read_text(encoding="utf-8"))
    if nested.exists():
        doc = json.loads(nested.read_text(encoding="utf-8"))
        return doc.get("scenarios") or doc
    return None


def main() -> int:
    log("─" * 60)
    log("AN03 — ANCHOR HARNESS REPRODUCTION TABLE")
    log("─" * 60)

    ours = _load_ours_summary()
    if ours is None:
        log(
            "  Note: results/ladder/anchor/scenario_summary.json missing — "
            "writing paper targets with ours=None (AN02 pending)."
        )

    rows = []
    for scen in cfg.ANCHOR["scenarios"]:
        sid = scen["id"]
        target = cfg.ANCHOR["table2_oracle"][sid]
        o = (ours or {}).get(sid) or {}
        label = scen["label"]

        rows.append(
            row(
                f"n_events_{sid}",
                f"N events — {label}",
                target["n_events"],
                o.get("n_events"),
                "paper Table 2",
                notes="Mean over 100 seeds in paper",
            )
        )
        rows.append(
            row(
                f"censor_pct_{sid}",
                f"Censoring % — {label}",
                target["censor_pct"],
                o.get("censor_pct"),
                "paper Table 2",
            )
        )
        rows.append(
            row(
                f"ci_oracle_{sid}",
                f"CI_oracle mean — {label}",
                target["ci_oracle_mean"],
                o.get("ci_oracle_mean"),
                "paper Table 2",
                notes=f"paper SD ±{target['ci_oracle_sd']}",
            )
        )
        rows.append(
            row(
                f"ibs_oracle_{sid}",
                f"IBS_oracle mean — {label}",
                target["ibs_oracle_mean"],
                o.get("ibs_oracle_mean"),
                "paper Table 2",
                notes=f"paper SD ±{target['ibs_oracle_sd']}",
                in_main_table=True,
            )
        )

    deviations = [
        "Harness check on synthetic DGP — not a fourth applied domain.",
        "Oracle metrics require true (e,c); unavailable on Domain 01–03.",
        f"Author code mirror: {cfg.ANCHOR['raw_dir'].relative_to(cfg.ROOT)}",
        "Figure 5 metric-error curves in results/ladder/anchor/figure5_metric_errors.json.",
    ]
    if ours is None:
        deviations.insert(
            2,
            "AN02 scenario_summary missing — ours_value blank until harness run.",
        )
    else:
        deviations.insert(
            2,
            "ours_value from AN02; use --full for paper n_seeds=100 before claiming Table 2 match.",
        )

    impl = "live" if ours else "targets_frozen_ours_pending"
    doc = build_document(
        domain_id="ANCHOR",
        baseline=f"Lillelund et al., {cfg.ANCHOR['venue']}",
        doi=cfg.ANCHOR["doi"],
        rows=rows,
        protocol_deviations=deviations,
        extra_meta={
            "role": "harness_check",
            "arxiv": cfg.ANCHOR["arxiv"],
            "code": cfg.ANCHOR["code"],
            "n_seeds_target": cfg.ANCHOR["n_seeds_default"],
            "implementation_status": impl,
        },
    )
    # Override paper_usage for harness framing
    doc["paper_usage"] = (
        "Harness-check asset: confirms our ladder/metric code recovers the "
        "anchor synthetic Table 2 (and later Figure 5 errors). Cite in Methods "
        "as validation of borrowed metrics — not as a Domain exhibit."
    )

    out_dir = cfg.DIRS["reproduction"]
    paths = write_reproduction_table(doc, out_dir)
    for k, p in paths.items():
        log(f"  Wrote {k:4s}: {p.relative_to(cfg.ROOT)}")

    status_line = (
        f"`ours` filled from AN02 ({impl})."
        if ours
        else "`ours` pending — run AN02."
    )
    asset = out_dir / "ANCHOR_paper_asset.md"
    asset.write_text(
        "\n".join(
            [
                "# Anchor harness check — paper asset",
                "",
                f"**Baseline:** {doc['baseline']} (`{cfg.ANCHOR['arxiv']}`)",
                "",
                "**Role:** calibrate Block E metrics on the authors' synthetic "
                "Weibull + Clayton experiment (Table 2 oracle CI/IBS).",
                "",
                f"**Status:** {status_line}",
                "",
                "Not Domain 04 — parallel to D lanes, feeds confidence in E.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    log(f"  Wrote {asset.relative_to(cfg.ROOT)}")
    log("AN03 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
