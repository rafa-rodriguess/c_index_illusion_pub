"""
AN05_ANCHOR_figure5_compare.py — Figure 5 censored-metric errors (paper vs ours)
==============================================================================
Compares metric errors (censored − oracle) that carry the ladder thesis:

  - bias_ci_harrell  (Harrell / naive CI error)
  - bias_ci_uno      (Uno IPCW CI error)
  - bias_uncens      (naive IBS error, no IPCW)
  - bias_ipcw        (IPCW IBS error)

Paper targets: executed summary embedded in author ``ladder_hypo.ipynb``
(Figure 5 source), frozen at ``data/processed/anchor/figure5_paper_targets.json``.

Ours: aggregated from ``results/ladder/anchor/seed_metrics.csv`` (AN02).

Writes:
  results/reproduction/ANCHOR_figure5_compare.{json,csv,md,tex}
  results/ladder/anchor/figure5_metric_errors.json  (refreshed mean±SD)

Execute:
    python -W default AN05_ANCHOR_figure5_compare.py
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

import pandas as pd

from src.anchor_harness import SETTING_TO_SCENARIO
from src.config import cfg

warnings.filterwarnings("default")

SCENARIO_ORDER = [
    ("random", "Random"),
    ("independent", "Independent"),
    ("dep_tau25", "Dependent (τ=0.25)"),
    ("dep_tau50", "Dependent (τ=0.50)"),
    ("dep_tau75", "Dependent (τ=0.75)"),
]

METRICS = [
    ("bias_ci_harrell", "CI Harrell (naive) error"),
    ("bias_ci_uno", "CI Uno (IPCW) error"),
    ("bias_uncens", "IBS naive (no IPCW) error"),
    ("bias_ipcw", "IBS IPCW error"),
]


def log(msg: str = "") -> None:
    print(msg, flush=True)


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _gap(ours: float | None, paper: float | None) -> float | None:
    if ours is None or paper is None:
        return None
    return float(ours) - float(paper)


def _map_setting(setting: Any) -> str | None:
    key = SETTING_TO_SCENARIO.get(setting)
    if key is None:
        try:
            key = SETTING_TO_SCENARIO.get(float(setting))
        except (TypeError, ValueError):
            key = None
    return key


def aggregate_ours(seed_csv: Path) -> dict[str, dict[str, float]]:
    df = pd.read_csv(seed_csv)
    out: dict[str, dict[str, float]] = {}
    for setting, g in df.groupby("setting"):
        sid = _map_setting(setting)
        if sid is None:
            continue
        row: dict[str, float] = {}
        for col, _ in METRICS:
            row[f"{col}_mean"] = float(g[col].mean())
            row[f"{col}_sd"] = float(g[col].std(ddof=1))
        out[sid] = row
    return out


def build_rows(paper: dict, ours: dict) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paper_sc = paper["scenarios"]
    for sid, label in SCENARIO_ORDER:
        p = paper_sc[sid]
        o = ours.get(sid) or {}
        for col, mlabel in METRICS:
            pm = p.get(f"{col}_mean")
            om = o.get(f"{col}_mean")
            rows.append(
                {
                    "scenario_id": sid,
                    "censoring": label,
                    "metric_id": col,
                    "metric": mlabel,
                    "paper_mean": pm,
                    "paper_sd": p.get(f"{col}_sd"),
                    "ours_mean": om,
                    "ours_sd": o.get(f"{col}_sd"),
                    "gap": _gap(om, pm),
                }
            )
    return rows


def _to_markdown(doc: dict[str, Any]) -> str:
    lines = [
        "# Anchor Figure 5 — censored metric errors (paper vs ours)",
        "",
        f"- Baseline: **{doc['baseline']}** (`{doc['arxiv']}`)",
        f"- Generated (UTC): `{doc['generated_at_utc']}`",
        f"- Paper targets: `{doc['paper_targets_source']}`",
        f"- Ours: AN02 `seed_metrics.csv` · n_seeds=`{doc.get('n_seeds_ours')}` · mode=`{doc.get('an02_mode')}`",
        f"- Error definition: `censored_metric − oracle` (paper footnote)",
        f"- Gap: `ours − paper`",
        "",
        "## Compact (mean error)",
        "",
        "| Censoring | Metric | Paper mean±SD | Ours mean±SD | Δ |",
        "|-----------|--------|--------------:|-------------:|--:|",
    ]
    for r in doc["rows"]:
        lines.append(
            f"| {r['censoring']} | {r['metric']} "
            f"| {_fmt(r['paper_mean'], 4)}±{_fmt(r['paper_sd'], 4)} "
            f"| {_fmt(r['ours_mean'], 4)}±{_fmt(r['ours_sd'], 4)} "
            f"| {_fmt(r['gap'], 4)} |"
        )

    # Wide pivot-style for CI and IBS separately
    by_sc: dict[str, dict[str, dict]] = {}
    for r in doc["rows"]:
        by_sc.setdefault(r["censoring"], {})[r["metric_id"]] = r

    lines += [
        "",
        "## CI errors (Figure 5a)",
        "",
        "| Censoring | Harrell paper | Harrell ours | ΔH | Uno paper | Uno ours | ΔU |",
        "|-----------|--------------:|-------------:|---:|----------:|---------:|---:|",
    ]
    for _, label in SCENARIO_ORDER:
        m = by_sc[label]
        h, u = m["bias_ci_harrell"], m["bias_ci_uno"]
        lines.append(
            f"| {label} "
            f"| {_fmt(h['paper_mean'], 4)} | {_fmt(h['ours_mean'], 4)} | {_fmt(h['gap'], 4)} "
            f"| {_fmt(u['paper_mean'], 4)} | {_fmt(u['ours_mean'], 4)} | {_fmt(u['gap'], 4)} |"
        )

    lines += [
        "",
        "## IBS errors (Figure 5b)",
        "",
        "| Censoring | Naive paper | Naive ours | ΔN | IPCW paper | IPCW ours | ΔI |",
        "|-----------|------------:|-----------:|---:|-----------:|----------:|---:|",
    ]
    for _, label in SCENARIO_ORDER:
        m = by_sc[label]
        n, i = m["bias_uncens"], m["bias_ipcw"]
        lines.append(
            f"| {label} "
            f"| {_fmt(n['paper_mean'], 4)} | {_fmt(n['ours_mean'], 4)} | {_fmt(n['gap'], 4)} "
            f"| {_fmt(i['paper_mean'], 4)} | {_fmt(i['ours_mean'], 4)} | {_fmt(i['gap'], 4)} |"
        )

    gaps = [abs(r["gap"]) for r in doc["rows"] if r["gap"] is not None]
    # Exclude Uno if flagged
    gaps_core = [
        abs(r["gap"])
        for r in doc["rows"]
        if r["gap"] is not None and r["metric_id"] != "bias_ci_uno"
    ]
    lines += [
        "",
        "## Verdict",
        "",
        f"- max |Δ| (all metrics) = `{_fmt(max(gaps) if gaps else None, 4)}`",
        f"- max |Δ| (excl. Uno) = `{_fmt(max(gaps_core) if gaps_core else None, 4)}`",
        "",
        doc.get("note", ""),
        "",
    ]
    return "\n".join(lines)


def _to_latex(doc: dict[str, Any]) -> str:
    lines = [
        "% Auto-generated ANCHOR Figure 5 compare",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{Anchor Figure 5: censored-metric errors vs oracle "
        r"(paper vs ours). Gap $=$ ours $-$ paper.}",
        r"  \label{tab:anchor-figure5-compare}",
        r"  \begin{tabular}{lrrrrrr}",
        r"    \toprule",
        r"    Censoring & Harrell$_p$ & Harrell$_o$ & $\Delta$H & "
        r"IPCW-IBS$_p$ & IPCW-IBS$_o$ & $\Delta$I \\",
        r"    \midrule",
    ]
    by_sc: dict[str, dict[str, dict]] = {}
    for r in doc["rows"]:
        by_sc.setdefault(r["censoring"], {})[r["metric_id"]] = r
    for _, label in SCENARIO_ORDER:
        m = by_sc[label]
        h, i = m["bias_ci_harrell"], m["bias_ipcw"]
        lab = label.replace("τ", r"$\tau$")
        lines.append(
            f"    {lab} & {_fmt(h['paper_mean'], 4)} & {_fmt(h['ours_mean'], 4)} & "
            f"{_fmt(h['gap'], 4)} & {_fmt(i['paper_mean'], 4)} & "
            f"{_fmt(i['ours_mean'], 4)} & {_fmt(i['gap'], 4)} \\\\"
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
    log("AN05 — ANCHOR FIGURE 5 COMPARE")
    log("─" * 60)

    targets_path = cfg.ANCHOR["processed_dir"] / "figure5_paper_targets.json"
    seed_csv = cfg.DIRS["ladder"] / "anchor" / "seed_metrics.csv"
    if not targets_path.exists():
        from src.repro import waiting_return

        return waiting_return(str(targets_path.relative_to(cfg.ROOT)))
    if not seed_csv.exists():
        from src.repro import waiting_return

        return waiting_return(f"{seed_csv.relative_to(cfg.ROOT)} (run AN02 --full).")

    paper = json.loads(targets_path.read_text(encoding="utf-8"))
    ours = aggregate_ours(seed_csv)
    meta_path = cfg.DIRS["ladder"] / "anchor" / "an02_run_manifest.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    rows = build_rows(paper, ours)
    n_seeds = int(meta.get("n_seeds") or len(pd.read_csv(seed_csv)["seed"].unique()))

    # Refresh ladder figure5 json with mean±SD
    fig5 = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_seeds": n_seeds,
        "note": "Metric error = censored metric − oracle (Figure 5)",
        "errors": {
            sid: {
                k: ours[sid][k]
                for k in ours[sid]
            }
            for sid in ours
        },
    }
    fig5_path = cfg.DIRS["ladder"] / "anchor" / "figure5_metric_errors.json"
    fig5_path.write_text(json.dumps(fig5, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    uno_gaps = [
        abs(r["gap"])
        for r in rows
        if r["metric_id"] == "bias_ci_uno" and r["gap"] is not None
    ]
    note = (
        "Figure 5 validates *censored* metrics (what Block E applies on real data). "
        "Paper targets = author notebook executed summary (not digitized from PDF). "
        "Harrell / IBS errors should match to ~1e-3 when AN02 uses author Uno path "
        "(sksurv IPCW). Larger Uno Δ indicates SurvivalEVAL-Uno vs sksurv mismatch."
    )
    if uno_gaps and max(uno_gaps) > 0.01:
        note += (
            f" Current max |ΔUno|={max(uno_gaps):.4f} — re-run "
            "`AN02_ANCHOR_run_ladder_hypo.py --full` after sksurv-Uno alignment."
        )

    doc: dict[str, Any] = {
        "artifact": "anchor_figure5_compare",
        "schema_version": "2026-07-11.anchor_figure5_compare.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": f"Lillelund et al., {cfg.ANCHOR['venue']}",
        "arxiv": cfg.ANCHOR["arxiv"],
        "doi": cfg.ANCHOR["doi"],
        "paper_targets_source": paper.get("source"),
        "an02_mode": meta.get("mode"),
        "n_seeds_ours": n_seeds,
        "gap_definition": "gap = ours_mean - paper_mean",
        "rows": rows,
        "note": note,
    }

    out_dir = cfg.DIRS["reproduction"]
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "ANCHOR_figure5_compare"

    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csv_path = out_dir / f"{stem}.csv"
    cols = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    md_path = out_dir / f"{stem}.md"
    md = _to_markdown(doc)
    md_path.write_text(md, encoding="utf-8")

    tex_path = out_dir / f"{stem}.tex"
    tex_path.write_text(_to_latex(doc), encoding="utf-8")

    for label, path in (
        ("json", json_path),
        ("csv", csv_path),
        ("md", md_path),
        ("tex", tex_path),
        ("fig5", fig5_path),
    ):
        log(f"  Wrote {label:4s}: {path.relative_to(cfg.ROOT)}")

    # Print CI + IBS sections
    parts = md.split("## CI errors")[1]
    log("\n## CI errors" + parts.split("## Verdict")[0])
    log("AN05 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
