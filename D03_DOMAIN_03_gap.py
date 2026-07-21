"""
D03_DOMAIN_03_gap.py — Paper-ready Table 8 reproduction asset
=============================================================
Writes ``results/reproduction/DOMAIN_03_reproduction_table.{json,csv,md,tex}``
and ``DOMAIN_03_paper_asset.md``.

Execute:
    python -W default D03_DOMAIN_03_gap.py
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


def main() -> int:
    log("─" * 60)
    log("D03_DOMAIN_03 — REPRODUCTION TABLE")
    log("─" * 60)

    metrics_path = cfg.DIRS["models"] / "domain3" / "rsf_metrics.json"
    if not metrics_path.exists():
        from src.repro import waiting_return

        return waiting_return("run D02 first.")

    m = json.loads(metrics_path.read_text(encoding="utf-8"))
    cells = m.get("cells") or {}

    rows = []
    for code in cfg.DOMAIN_03["communities"]:
        label = cfg.DOMAIN_03["community_labels"][code]
        for fset in ("behavioural", "content", "combined"):
            for theta in cfg.DOMAIN_03["theta_months"]:
                key = f"{code}|{fset}|{theta}"
                cell = cells.get(key) or {}
                paper = cfg.DOMAIN_03["paper_table8"].get((code, fset, int(theta)))
                rows.append(
                    row(
                        f"cindex_{code}_{fset}_theta{theta}",
                        f"C-index — {label} / {fset} / θ={theta}",
                        paper,
                        cell.get("cindex_mean"),
                        "paper Table 8",
                        notes=f"ours STD={cell.get('cindex_std')}; n_scores={cell.get('n_scores')}",
                        in_main_table=True,
                    )
                )

    # Band coverage
    ours_vals = [r["ours_value"] for r in rows if r["ours_value"] is not None]
    band = cfg.DOMAIN_03["target_cindex_band"]
    in_band = sum(1 for v in ours_vals if band[0] <= v <= band[1])
    rows.append(
        row(
            "band_coverage",
            f"Cells with C in [{band[0]}, {band[1]}]",
            len(cfg.DOMAIN_03["paper_table8"]),
            in_band,
            "roadmap / Table 8 band",
            notes=f"{in_band}/{len(ours_vals)} ours cells inside global band",
            in_main_table=False,
        )
    )

    doc = build_document(
        domain_id="DOMAIN_03",
        baseline=cfg.DOMAIN_03["baseline"],
        doi=cfg.DOMAIN_03["doi"],
        rows=rows,
        protocol_deviations=list(m.get("protocol_deviations") or []),
        extra_meta={
            "fase_a_status": "complete",
            "github": cfg.DOMAIN_03["github"],
            "rsf": m.get("rsf_hyperparameters"),
            "cv": m.get("cv"),
            "backend": m.get("backend"),
        },
    )
    paths = write_reproduction_table(doc, cfg.ROOT / "results" / "reproduction")

    # Headline paper asset
    lines = [
        "# DOMAIN_03 — paper asset (Fase A)",
        "",
        f"- Baseline: **{cfg.DOMAIN_03['baseline']}**",
        f"- DOI: `{cfg.DOMAIN_03['doi']}`",
        f"- GitHub: {cfg.DOMAIN_03['github']}",
        f"- Full table: `DOMAIN_03_reproduction_table.{{json,csv,md,tex}}`",
        "",
        "## Table 8 — mean C-index (paper vs ours)",
        "",
        "| Community | Features | θ | Paper | Ours | Gap |",
        "|-----------|----------|--:|------:|-----:|----:|",
    ]
    for code in cfg.DOMAIN_03["communities"]:
        label = cfg.DOMAIN_03["community_labels"][code]
        for fset in ("behavioural", "content", "combined"):
            for theta in cfg.DOMAIN_03["theta_months"]:
                cell = cells.get(f"{code}|{fset}|{theta}") or {}
                paper = cfg.DOMAIN_03["paper_table8"].get((code, fset, int(theta)))
                ours = cell.get("cindex_mean")
                gap = (ours - paper) if ours is not None and paper is not None else None
                lines.append(
                    f"| {label} | {fset} | {theta} | {paper:.2f} | "
                    f"{ours:.3f} | {gap:+.3f} |"
                )
    lines += [
        "",
        "## Notes",
        "",
        "- RSF: sksurv with author-notebook hyperparameters (5 trees, depth 5, leaf 30).",
        "- CV: 5-fold × 30 runs.",
        "- Deviations listed in `DOMAIN_03_reproduction_table.md`.",
        "",
    ]
    paper_md = cfg.ROOT / "results" / "reproduction" / "DOMAIN_03_paper_asset.md"
    paper_md.write_text("\n".join(lines), encoding="utf-8")

    summary = {
        "stage": "D03_DOMAIN_03",
        "domain_id": "DOMAIN_03",
        "fase_a_status": "complete",
        "artifact": {k: str(p.relative_to(cfg.ROOT)) for k, p in paths.items()},
        "paper_asset_md": str(paper_md.relative_to(cfg.ROOT)),
        "n_cells": len(cells),
        "mean_abs_gap": float(
            np_mean_abs_gap(cells)
        ),
        "protocol_deviations": doc["protocol_deviations"],
    }
    (cfg.ROOT / "results" / "reproduction" / "domain3_gap.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    for k, p in paths.items():
        log(f"  Wrote {k}: {p.relative_to(cfg.ROOT)}")
    log(f"  Wrote paper asset: {paper_md.relative_to(cfg.ROOT)}")
    log("D03_DOMAIN_03 complete.")
    return 0


def np_mean_abs_gap(cells: dict) -> float:
    gaps = [abs(v["gap"]) for v in cells.values() if v.get("gap") is not None]
    return sum(gaps) / len(gaps) if gaps else float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
