"""
AN04_ANCHOR_compare_table.py — Compact Table-2 compare (paper vs ours)
=====================================================================
Reads AN02 ``scenario_summary_flat.json`` + ``cfg.ANCHOR["table2_oracle"]``
and writes a side-by-side artifact:

    results/reproduction/ANCHOR_table2_compare.{json,csv,md,tex}

One row per censoring scenario (paper Table 2 layout).

Prereq: AN02 ``--full`` (or any run that wrote scenario_summary_flat.json).

Execute:
    python -W default AN04_ANCHOR_compare_table.py
"""

from __future__ import annotations

import csv
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import cfg

warnings.filterwarnings("default")

SCENARIO_ORDER = [
    ("random", "Random"),
    ("independent", "Independent"),
    ("dep_tau25", "Dependent (τ=0.25)"),
    ("dep_tau50", "Dependent (τ=0.50)"),
    ("dep_tau75", "Dependent (τ=0.75)"),
]


def log(msg: str = "") -> None:
    print(msg, flush=True)


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def _gap(ours: float | None, paper: float | None) -> float | None:
    if ours is None or paper is None:
        return None
    return float(ours) - float(paper)


def _load_ours() -> dict[str, Any] | None:
    flat = cfg.DIRS["ladder"] / "anchor" / "scenario_summary_flat.json"
    nested = cfg.DIRS["ladder"] / "anchor" / "scenario_summary.json"
    if flat.exists():
        return json.loads(flat.read_text(encoding="utf-8"))
    if nested.exists():
        doc = json.loads(nested.read_text(encoding="utf-8"))
        return doc.get("scenarios") or doc
    return None


def _load_run_meta() -> dict[str, Any]:
    man = cfg.DIRS["ladder"] / "anchor" / "an02_run_manifest.json"
    if man.exists():
        return json.loads(man.read_text(encoding="utf-8"))
    return {}


def build_rows(ours: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paper = cfg.ANCHOR["table2_oracle"]
    for sid, label in SCENARIO_ORDER:
        p = paper[sid]
        o = (ours or {}).get(sid) or {}
        rows.append(
            {
                "scenario_id": sid,
                "censoring": label,
                "n_events_paper": p["n_events"],
                "n_events_ours": o.get("n_events"),
                "n_events_gap": _gap(o.get("n_events"), p["n_events"]),
                "censor_pct_paper": p["censor_pct"],
                "censor_pct_ours": o.get("censor_pct"),
                "censor_pct_gap": _gap(o.get("censor_pct"), p["censor_pct"]),
                "ci_oracle_paper": p["ci_oracle_mean"],
                "ci_oracle_paper_sd": p["ci_oracle_sd"],
                "ci_oracle_ours": o.get("ci_oracle_mean"),
                "ci_oracle_ours_sd": o.get("ci_oracle_sd"),
                "ci_oracle_gap": _gap(o.get("ci_oracle_mean"), p["ci_oracle_mean"]),
                "ibs_oracle_paper": p["ibs_oracle_mean"],
                "ibs_oracle_paper_sd": p["ibs_oracle_sd"],
                "ibs_oracle_ours": o.get("ibs_oracle_mean"),
                "ibs_oracle_ours_sd": o.get("ibs_oracle_sd"),
                "ibs_oracle_gap": _gap(o.get("ibs_oracle_mean"), p["ibs_oracle_mean"]),
            }
        )
    return rows


def _to_markdown(doc: dict[str, Any]) -> str:
    lines = [
        "# Anchor Table 2 — paper vs ours",
        "",
        f"- Baseline: **{doc['baseline']}** (`{doc['arxiv']}`)",
        f"- Generated (UTC): `{doc['generated_at_utc']}`",
        f"- AN02 mode: `{doc.get('an02_mode')}` · seeds: `{doc.get('n_seeds')}`",
        f"- Gap: `ours − paper`",
        "",
        "## Side-by-side",
        "",
        "| Censoring | N events (paper) | N events (ours) | ΔN | "
        "Cens% (paper) | Cens% (ours) | Δ% | "
        "CI_oracle (paper) | CI_oracle (ours) | ΔCI | "
        "IBS_oracle (paper) | IBS_oracle (ours) | ΔIBS |",
        "|-----------|-----------------:|----------------:|---:|"
        "-------------:|-------------:|---:|"
        "-----------------:|----------------:|----:|"
        "------------------:|-----------------:|-----:|",
    ]
    for r in doc["rows"]:
        lines.append(
            f"| {r['censoring']} "
            f"| {_fmt(r['n_events_paper'], 0)} "
            f"| {_fmt(r['n_events_ours'], 1)} "
            f"| {_fmt(r['n_events_gap'], 2)} "
            f"| {_fmt(r['censor_pct_paper'], 1)} "
            f"| {_fmt(r['censor_pct_ours'], 2)} "
            f"| {_fmt(r['censor_pct_gap'], 3)} "
            f"| {_fmt(r['ci_oracle_paper'], 3)} "
            f"| {_fmt(r['ci_oracle_ours'], 4)} "
            f"| {_fmt(r['ci_oracle_gap'], 4)} "
            f"| {_fmt(r['ibs_oracle_paper'], 3)} "
            f"| {_fmt(r['ibs_oracle_ours'], 4)} "
            f"| {_fmt(r['ibs_oracle_gap'], 4)} |"
        )

    # Compact verdict table
    lines += [
        "",
        "## Compact (metrics only)",
        "",
        "| Censoring | CI_oracle paper | CI_oracle ours | ΔCI | "
        "IBS_oracle paper | IBS_oracle ours | ΔIBS |",
        "|-----------|----------------:|---------------:|----:|"
        "-----------------:|----------------:|-----:|",
    ]
    for r in doc["rows"]:
        lines.append(
            f"| {r['censoring']} "
            f"| {_fmt(r['ci_oracle_paper'], 3)}±{_fmt(r['ci_oracle_paper_sd'], 3)} "
            f"| {_fmt(r['ci_oracle_ours'], 4)} "
            f"| {_fmt(r['ci_oracle_gap'], 4)} "
            f"| {_fmt(r['ibs_oracle_paper'], 3)}±{_fmt(r['ibs_oracle_paper_sd'], 3)} "
            f"| {_fmt(r['ibs_oracle_ours'], 4)} "
            f"| {_fmt(r['ibs_oracle_gap'], 4)} |"
        )

    max_abs_ci = max(
        (abs(r["ci_oracle_gap"]) for r in doc["rows"] if r["ci_oracle_gap"] is not None),
        default=None,
    )
    max_abs_ibs = max(
        (abs(r["ibs_oracle_gap"]) for r in doc["rows"] if r["ibs_oracle_gap"] is not None),
        default=None,
    )
    lines += [
        "",
        "## Verdict",
        "",
        f"- max |ΔCI_oracle| = `{_fmt(max_abs_ci, 4)}`",
        f"- max |ΔIBS_oracle| = `{_fmt(max_abs_ibs, 4)}`",
        "- Tier (C00.4): ≤0.01 strict for discrimination-style gaps.",
        "",
        doc.get("note", ""),
        "",
    ]
    return "\n".join(lines)


def _to_latex(doc: dict[str, Any]) -> str:
    lines = [
        "% Auto-generated ANCHOR Table 2 compare — do not edit by hand",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Anchor harness check: Table 2 oracle metrics "
        r"(paper vs ours). Gap $=$ ours $-$ paper.}",
        r"  \label{tab:anchor-table2-compare}",
        r"  \begin{tabular}{lrrrrrr}",
        r"    \toprule",
        r"    Censoring & CI$_\mathrm{paper}$ & CI$_\mathrm{ours}$ & "
        r"$\Delta$CI & IBS$_\mathrm{paper}$ & IBS$_\mathrm{ours}$ & "
        r"$\Delta$IBS \\",
        r"    \midrule",
    ]
    for r in doc["rows"]:
        label = str(r["censoring"]).replace("τ", r"$\tau$")
        lines.append(
            f"    {label} & {_fmt(r['ci_oracle_paper'], 3)} & "
            f"{_fmt(r['ci_oracle_ours'], 4)} & {_fmt(r['ci_oracle_gap'], 4)} & "
            f"{_fmt(r['ibs_oracle_paper'], 3)} & {_fmt(r['ibs_oracle_ours'], 4)} & "
            f"{_fmt(r['ibs_oracle_gap'], 4)} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    log("─" * 60)
    log("AN04 — ANCHOR TABLE 2 COMPARE")
    log("─" * 60)

    ours = _load_ours()
    if ours is None:
        from src.repro import waiting_return

        return waiting_return(
            "results/ladder/anchor/scenario_summary_flat.json (run AN02)."
        )

    meta = _load_run_meta()
    rows = build_rows(ours)
    doc: dict[str, Any] = {
        "artifact": "anchor_table2_compare",
        "schema_version": "2026-07-11.anchor_table2_compare.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": f"Lillelund et al., {cfg.ANCHOR['venue']}",
        "arxiv": cfg.ANCHOR["arxiv"],
        "doi": cfg.ANCHOR["doi"],
        "an02_mode": meta.get("mode"),
        "n_seeds": meta.get("n_seeds"),
        "gap_definition": "gap = ours - paper",
        "rows": rows,
        "note": (
            "Harness validation of borrowed ladder metrics on the authors' "
            "synthetic Weibull+Clayton experiment. Not Domain 04."
        ),
    }

    out_dir = cfg.DIRS["reproduction"]
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "ANCHOR_table2_compare"

    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csv_cols = list(rows[0].keys()) if rows else []
    csv_path = out_dir / f"{stem}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=csv_cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    md_path = out_dir / f"{stem}.md"
    md_path.write_text(_to_markdown(doc), encoding="utf-8")

    tex_path = out_dir / f"{stem}.tex"
    tex_path.write_text(_to_latex(doc), encoding="utf-8")

    for label, path in (
        ("json", json_path),
        ("csv", csv_path),
        ("md", md_path),
        ("tex", tex_path),
    ):
        log(f"  Wrote {label:4s}: {path.relative_to(cfg.ROOT)}")

    # Print compact table to stdout for chat / CI
    log("")
    log(_to_markdown(doc).split("## Compact")[1].split("## Verdict")[0].strip())
    log("AN04 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
