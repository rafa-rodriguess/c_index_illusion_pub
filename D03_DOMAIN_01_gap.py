"""
D03_DOMAIN_01_gap.py — Paper-ready mixed reproduction table (DOMAIN_01)
=======================================================================
Builds the canonical mixed artifact:

    results/reproduction/DOMAIN_01_reproduction_table.{json,csv,md,tex}

Rows = quantities extracted from Ahmed & Green (2024) side-by-side with
our pipeline values and gap (ours − paper). Also writes a compact
``domain1_gap.json`` summary for D99 merge.

Prereq: D00 decisions + D02 ``cox_metrics.json``.

Execute:
    python -W default D03_DOMAIN_01_gap.py
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


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    log("─" * 60)
    log("D03_DOMAIN_01 — REPRODUCTION TABLE (paper asset)")
    log("─" * 60)

    out_dir = cfg.ROOT / "results" / "reproduction"
    metrics_path = cfg.DIRS["models"] / "domain1" / "cox_metrics.json"
    decisions_path = cfg.DIRS["logs"] / "d00_domain_01_decisions.json"

    m = _load_json(metrics_path)
    decisions = _load_json(decisions_path) or {}
    counts = decisions.get("counts") or {}

    if m is None:
        from src.repro import waiting_return

        return waiting_return(f"{metrics_path.relative_to(cfg.ROOT)} (run D02).")

    hr = m.get("paper_hr_highlights") or {}
    hr184 = hr.get("smart_184_raw") or {}
    hr190 = hr.get("smart_190_raw") or {}
    hr194 = hr.get("smart_194_raw") or {}

    rows = [
        row(
            "n_drives_st4000dm000",
            "N drives (ST4000DM000, 2013–2022)",
            cfg.DOMAIN_01["n_drives_reported"],
            counts.get("n_drives_total"),
            "paper §4",
            notes="Population filter",
        ),
        row(
            "n_healthy_age_gt7",
            "N healthy in Cox H6a sample (calendar>7y)",
            cfg.DOMAIN_01["cox_cohort_reported"]["healthy"],
            counts.get("n_cox_healthy"),
            "paper §4.1",
            notes="H6a: calendar_span_years>7 healthy ∪ all failed (not SMART9-both-classes)",
        ),
        row(
            "n_failed_age_gt7",
            "N failed in Cox H6a sample (all failed)",
            cfg.DOMAIN_01["cox_cohort_reported"]["failed"],
            counts.get("n_cox_failed"),
            "paper §4.1",
            notes="H6a keeps all failures; paper wording ambiguous on failed age filter",
        ),
        row(
            "harrell_cindex",
            "Harrell C-index (Cox PH)",
            cfg.DOMAIN_01["target_value"],
            m.get("harrell_cindex"),
            "paper §7.1 / abstract",
            notes="In-sample GOF on H6a; lifelines.CoxPHFitter",
        ),
        row(
            "hr_smart_184",
            "Hazard ratio SMART 184",
            hr184.get("paper_hr", 1.010),
            hr184.get("ours_hr"),
            "paper §7.1",
        ),
        row(
            "hr_smart_190",
            "Hazard ratio SMART 190",
            hr190.get("paper_hr", 0.984),
            hr190.get("ours_hr"),
            "paper §7.1",
        ),
        row(
            "hr_smart_194",
            "Hazard ratio SMART 194",
            hr194.get("paper_hr", 0.990),
            hr194.get("ours_hr"),
            "paper §7.1",
            notes=hr194.get("note") or "Dropped: identical to SMART 190 on this model",
            in_main_table=False,
        ),
        row(
            "n_fit_complete_cases",
            "N drives in Cox fit (complete cases)",
            None,
            m.get("n_fit"),
            "ours only",
            notes=f"events={m.get('n_events')}; penalizer={m.get('penalizer')}",
            in_main_table=False,
        ),
    ]

    doc = build_document(
        domain_id="DOMAIN_01",
        baseline=cfg.DOMAIN_01["baseline"],
        doi=cfg.DOMAIN_01["doi"],
        rows=rows,
        protocol_deviations=list(m.get("protocol_deviations") or []),
        extra_meta={
            "eval_mode": m.get("eval_mode"),
            "fit_population": m.get("fit_population"),
            "author_code_url": cfg.DOMAIN_01.get("author_code_url"),
            "author_code_status": cfg.DOMAIN_01.get("author_code_status"),
            "model_path": m.get("model_path"),
        },
    )

    paths = write_reproduction_table(doc, out_dir)

    # Compact summary for D99 / quick checks (legacy-friendly name)
    c_row = next(r for r in rows if r["quantity_id"] == "harrell_cindex")
    summary = {
        "stage": "D03_DOMAIN_01",
        "domain_id": "DOMAIN_01",
        "baseline": cfg.DOMAIN_01["baseline"],
        "doi": cfg.DOMAIN_01["doi"],
        "artifact": {k: str(p.relative_to(cfg.ROOT)) for k, p in paths.items()},
        "headline": {
            "metric": "harrell_cindex",
            "paper": c_row["paper_value"],
            "ours": c_row["ours_value"],
            "gap": c_row["gap"],
        },
        "status": "ok" if c_row["ours_value"] is not None else "pending",
        "protocol_deviations": doc["protocol_deviations"],
    }
    summary_path = out_dir / "domain1_gap.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    log(f"  Headline C  paper={c_row['paper_value']}  ours={c_row['ours_value']}  gap={c_row['gap']}")
    for k, p in paths.items():
        log(f"  Wrote {k:4s}: {p.relative_to(cfg.ROOT)}")
    log(f"  Wrote summary: {summary_path.relative_to(cfg.ROOT)}")
    log("D03_DOMAIN_01 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
