"""
G02_final_verdict_table.py — Block G: paper verdict table
=========================================================
Authoritative decisions come from **G03** (outer Holm + H_meta).

Also pulls:
  - results/probes/H01_ranking_inversion.json   (H1 descriptive τ_K)
  - results/probes/H04_paired_bootstrap_loo.json (H4 formal protocol)
  - results/probes/report.json                  (F03 one-liners)
  - results/ladder/summary.json                 (ladder status labels)
  - G00 / G01 process controls
  - DOMAIN_0n gap pointers

Writes:
  results/reproduction/FINAL_verdict_table.{json,md,tex,csv}
  results/logs/final_verdict.md

Execute (after G03):
    python -W default G02_final_verdict_table.py
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import cfg
from src.metrics.io import write_json


def log(msg: str = "") -> None:
    print(msg, flush=True)


def _read(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _gap_line(domain: str) -> str:
    alt = {
        "DOMAIN_01": cfg.DIRS["reproduction"] / "domain1_gap.json",
        "DOMAIN_02": cfg.DIRS["reproduction"] / "domain2_gap.json",
        "DOMAIN_03": cfg.DIRS["reproduction"] / "domain3_gap.json",
    }.get(domain)
    doc = _read(alt) if alt else None
    if not doc:
        return "—"
    if doc.get("headline"):
        return str(doc["headline"])[:160]
    if doc.get("mean_abs_gap") is not None:
        return f"mean_abs_gap={doc['mean_abs_gap']} (n_cells={doc.get('n_cells')})"
    if "gaps" in doc and isinstance(doc["gaps"], list) and doc["gaps"]:
        g0 = doc["gaps"][0]
        return f"{g0.get('quantity_id')}: gap={g0.get('gap')}"
    return "see DOMAIN_*_reproduction_table.md"


def _status_label(flag: dict[str, Any] | None) -> str:
    if not flag:
        return "missing"
    return str(flag.get("status") or "unknown")


def main() -> int:
    log("─" * 60)
    log("G02 — FINAL VERDICT TABLE (G03 = formal source)")
    log("─" * 60)

    g03 = _read(cfg.DIRS["reproduction"] / "G03_holm_family.json")
    if not g03:
        log("ERROR: missing G03_holm_family.json — run G03_holm_family.py first")
        return 1

    ladder = _read(cfg.DIRS["ladder"] / "summary.json") or {}
    probes = _read(cfg.DIRS["probes"] / "report.json") or {}
    h01 = _read(cfg.DIRS["probes"] / "H01_ranking_inversion.json") or {}
    h04 = _read(cfg.DIRS["probes"] / "H04_paired_bootstrap_loo.json") or {}
    g00 = _read(cfg.DIRS["logs"] / "leakage_controls_audit.json") or {}
    g01 = _read(cfg.ROOT / "results" / "harness" / "manifest.json") or {}

    flags = ladder.get("hypothesis_flags") or {}
    outer = g03.get("outer_holm") or {}
    raw_p = g03.get("raw_p") or outer.get("raw") or {}
    adj_p = outer.get("adjusted") or {}
    holm_rej = outer.get("reject") or {}
    protocol_rej = g03.get("protocol_reject") or {}
    details = g03.get("details") or {}
    h_meta = g03.get("H_meta") or {}

    def _note(hid: str) -> str:
        if hid == "H1":
            inv = h01.get("any_binary_inversion")
            formal = h01.get("formal_reject_C00_1_any_domain")
            d2 = next(
                (d for d in (h01.get("domains") or []) if d.get("domain_id") == "DOMAIN_02"),
                {},
            )
            tau = d2.get("tau_K")
            ci = (d2.get("bootstrap") or {}).get("ci")
            return (
                f"D2 τ_K={tau}; inversion={inv}; boot CI={ci}; "
                f"C00.1 v5 formal={formal}; Holm adj p={adj_p.get('H1')}"
            )
        if hid == "H4":
            return (
                (probes.get("H4") or {}).get("one_liner")
                or f"H04 h4_reject={h04.get('h4_reject')} "
                f"({h04.get('n_hits_reject')}/{h04.get('n_hits')} LOO hits)"
            )
        return (
            (probes.get(hid) or {}).get("one_liner")
            or (flags.get(hid) or {}).get("note")
            or (details.get(hid) or {}).get("source")
            or "—"
        )

    titles = {
        "H1": "Ranking inversion (C vs IBS)",
        "H2": "High C fails D-Calibration",
        "H3": "Competing-risks blindness",
        "H4": "Broad SMART ablation (existential)",
        "H5": "Brier↑ under stable C",
    }
    domains = {
        "H1": "all (C00.2)",
        "H2": "Backblaze",
        "H3": "Bondora",
        "H4": "Backblaze",
        "H5": "Stack Exchange",
    }
    stats = {
        "H1": "Kendall τ_K + subject bootstrap",
        "H2": "D-Cal p (SurvivalEVAL)",
        "H3": "|F_naive−F_AJ| @12m",
        "H4": "∃ ablation ΔC_H",
        "H5": "Brier(36)−Brier(12)",
    }
    thresholds = {
        "H1": "τ_K≤0.5 AND boot p(τ=1)<α (C00.1 v5)",
        "H2": "C_H≥0.90 and p<0.05",
        "H3": ">0.02, CI∌0, ≥3 strata",
        "H4": "≥0.03 + non-overlapping CIs",
        "H5": "mono↑, >2 SE, C∈[0.66,0.76]",
    }

    rows = []
    for hid in ("H1", "H2", "H3", "H4", "H5"):
        rows.append(
            {
                "id": hid,
                "title": titles[hid],
                "domain": domains[hid],
                "statistic": stats[hid],
                "threshold": thresholds[hid],
                "ladder_status": _status_label(flags.get(hid)),
                "raw_p": raw_p.get(hid),
                "holm_adjusted_p": adj_p.get(hid),
                "holm_reject": bool(holm_rej.get(hid)),
                "protocol_reject": bool(protocol_rej.get(hid)),
                "result_note": _note(hid),
            }
        )

    n_holm = sum(1 for r in rows if r["holm_reject"])
    meta_min = int(h_meta.get("min_rejected_of_five") or cfg.PROTOCOL["hypotheses"]["H_meta"]["min_rejected_of_five"])
    h_meta_reject = bool(h_meta.get("reject"))

    reproduction = {
        "DOMAIN_01": _gap_line("DOMAIN_01"),
        "DOMAIN_02": _gap_line("DOMAIN_02"),
        "DOMAIN_03": _gap_line("DOMAIN_03"),
    }

    doc = {
        "stage": "G02",
        "artifact": "final_verdict_table",
        "authority": "G03_holm_family",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": cfg.PROTOCOL.get("version"),
        "g03_generated_at_utc": g03.get("generated_at_utc"),
        "hypotheses": rows,
        "n_holm_reject": n_holm,
        "H_meta": {
            "min_rejected_of_five": meta_min,
            "n_outer_rejects": h_meta.get("n_outer_rejects", n_holm),
            "reject": h_meta_reject,
            "rejected_hypotheses": h_meta.get("rejected_hypotheses")
            or [r["id"] for r in rows if r["holm_reject"]],
            "rule": h_meta.get("rule")
            or f"Reject H_meta if ≥{meta_min} of 5 outer-Holm rejects",
            "note": (
                "Formal — outer Holm from G03. "
                f"H1 C00.1 v5 formal_reject_any={h01.get('formal_reject_C00_1_any_domain')}."
            ),
        },
        "g00_audit": {
            "overall_status": g00.get("overall_status"),
            "n_pass": g00.get("n_pass"),
            "n_warn": g00.get("n_warn"),
            "n_fail": g00.get("n_fail"),
        },
        "g01_harness": {
            "export_root": g01.get("export_root"),
            "protocol_version": g01.get("protocol_version"),
            "n_modules": len(g01.get("modules") or []),
        },
        "reproduction_gaps": reproduction,
        "honesty": [
            "Decisions in this table are G03 outer-Holm rejects (formal). Ladder/probe flags are inputs only.",
            (
                "H1: C00.1 v5 = subject-level bootstrap p(τ=1)<α with τ≤0.5; "
                f"formal_reject_any={h01.get('formal_reject_C00_1_any_domain')}; "
                f"B={h01.get('B')}."
            ),
            (
                f"H4: holm_reject={bool(holm_rej.get('H4'))}; "
                f"H04 h4_reject={h04.get('h4_reject')}; "
                f"n_hits={h04.get('n_hits')} / n_hits_reject={h04.get('n_hits_reject')} "
                f"(B={h04.get('bootstrap_B')})."
            ),
            f"PROTOCOL {cfg.PROTOCOL.get('version')} — H1 subject-bootstrap; H4 existential LOO; H5 Brier-primary.",
        ],
    }

    out = cfg.DIRS["reproduction"]
    out.mkdir(parents=True, exist_ok=True)
    stem = out / "FINAL_verdict_table"
    write_json(Path(str(stem) + ".json"), doc)

    md = [
        "# Final verdict table (Block G / roadmap §10.2)",
        "",
        f"_Generated UTC: `{doc['generated_at_utc']}` · protocol `{doc['protocol_version']}`_",
        "",
        "> **Formal source: G03 outer Holm.** Ladder/probe flags are inputs only — not the decision.",
        "",
        "## Hypotheses H1–H5",
        "",
        "| # | Title | Domain | Statistic | Threshold | Holm reject | adj p | Result |",
        "|---|-------|--------|-----------|-----------|-------------|-------|--------|",
    ]
    for r in rows:
        adj = r["holm_adjusted_p"]
        adj_s = f"{adj:.4g}" if isinstance(adj, float) else "—"
        md.append(
            f"| {r['id']} | {r['title']} | {r['domain']} | `{r['statistic']}` | "
            f"{r['threshold']} | **{r['holm_reject']}** | {adj_s} | "
            f"{(r.get('result_note') or '—')[:110]} |"
        )
    md += [
        "",
        "## H_meta (formal)",
        "",
        f"- Outer-Holm rejects among H1–H5: **{n_holm}** / 5",
        f"- Threshold: ≥**{meta_min}**",
        f"- **H_meta reject:** **`{h_meta_reject}`**",
        f"- Rejected: {', '.join(doc['H_meta']['rejected_hypotheses']) or '—'}",
        f"- Note: {doc['H_meta']['note']}",
        "",
        "## Process controls",
        "",
        f"- G00 leakage audit: `{doc['g00_audit'].get('overall_status')}` "
        f"(pass={doc['g00_audit'].get('n_pass')}, warn={doc['g00_audit'].get('n_warn')}, "
        f"fail={doc['g00_audit'].get('n_fail')})",
        f"- G01 harness export: `{doc['g01_harness'].get('export_root')}` "
        f"({doc['g01_harness'].get('n_modules')} modules)",
        f"- G03 authority: `results/reproduction/G03_holm_family.*` "
        f"(generated `{doc.get('g03_generated_at_utc')}`)",
        "",
        "## Reproduction gaps (pointers)",
        "",
        "| Domain | Gap pointer |",
        "|--------|-------------|",
        f"| DOMAIN_01 | {reproduction['DOMAIN_01']} |",
        f"| DOMAIN_02 | {reproduction['DOMAIN_02']} |",
        f"| DOMAIN_03 | {reproduction['DOMAIN_03']} |",
        "",
        "## Honesty",
        "",
    ]
    for h in doc["honesty"]:
        md.append(f"- {h}")
    md.append("")
    Path(str(stem) + ".md").write_text("\n".join(md), encoding="utf-8")

    tex = [
        "% Auto-generated by G02_final_verdict_table.py (authority=G03)",
        "\\begin{tabular}{llllll}",
        "\\toprule",
        "ID & Domain & Statistic & Holm reject & adj $p$ & Note \\\\",
        "\\midrule",
    ]
    for r in rows:
        note = (r.get("result_note") or "").replace("&", "\\&")[:55]
        adj = r["holm_adjusted_p"]
        adj_s = f"{adj:.3g}" if isinstance(adj, float) else "—"
        tex.append(
            f"{r['id']} & {r['domain']} & {r['statistic']} & "
            f"{r['holm_reject']} & {adj_s} & {note} \\\\"
        )
    tex += [
        "\\midrule",
        f"H\\_meta & — & $\\ge {meta_min}/5$ & {h_meta_reject} & — & "
        f"n={n_holm} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ]
    Path(str(stem) + ".tex").write_text("\n".join(tex), encoding="utf-8")

    with Path(str(stem) + ".csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "id",
            "title",
            "domain",
            "statistic",
            "threshold",
            "ladder_status",
            "raw_p",
            "holm_adjusted_p",
            "holm_reject",
            "protocol_reject",
            "result_note",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    pointer = [
        "# Final verdict (pointer)",
        "",
        "See `results/reproduction/FINAL_verdict_table.md`.",
        f"Authority=G03; H_meta reject={h_meta_reject}; Holm rejects={n_holm}/5.",
        f"G00={doc['g00_audit'].get('overall_status')}; "
        f"G01={doc['g01_harness'].get('export_root')}.",
        "",
    ]
    (cfg.DIRS["logs"] / "final_verdict.md").write_text("\n".join(pointer), encoding="utf-8")

    log(f"  Holm rejects: {n_holm}/5  H_meta={h_meta_reject}")
    log(f"  G00={doc['g00_audit'].get('overall_status')}  G01={doc['g01_harness'].get('export_root')}")
    for r in rows:
        log(f"  {r['id']}: holm_reject={r['holm_reject']}  adj_p={r['holm_adjusted_p']}")
    log(f"  Wrote {stem.relative_to(cfg.ROOT)}.*")
    log("G02 complete.")

    if g00.get("overall_status") == "fail":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
