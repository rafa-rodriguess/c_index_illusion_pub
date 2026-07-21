"""
F01_competing_risks_bias_bondora.py — H3 probe (CR blindness exhibit)
====================================================================
Paper-facing Bondora exhibit: naive default CIF (early repayment as censoring)
vs Aalen–Johansen CIF (early as competitor) at 12 months.

Reuses ``src.metrics.competing_risks`` (same engine as E05). Writes:

  results/probes/F01_h3_competing_risks.{json,md,tex,csv}

Execute:
    python -W default F01_competing_risks_bias_bondora.py
"""

from __future__ import annotations

import csv
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import cfg
from src.metrics.competing_risks import build_cr_frame, cif_bias, cif_bias_by_rating
from src.metrics.io import write_json

warnings.filterwarnings("default")
# lifelines AJ jitters ties and warns once per bootstrap draw — drown the log
warnings.filterwarnings(
    "ignore",
    message=r".*Tied event times were detected.*",
    module=r"lifelines\.fitters\.aalen_johansen_fitter",
)


def log(msg: str = "") -> None:
    print(msg, flush=True)


def main() -> int:
    log("─" * 60)
    log("F01 — H3 COMPETING-RISKS BIAS (Bondora)")
    log("─" * 60)

    path = cfg.ROOT / "data/processed/domain2/loans_split.parquet"
    if not path.exists():
        log(f"ERROR: missing {path}")
        return 1

    h3 = cfg.PROTOCOL["hypotheses"]["H3"]
    delta_thr = float(h3["delta"])
    horizon_m = float(h3["horizon_months"])
    min_strata = int(h3["min_strata_consistent"])
    B = min(int(cfg.EVAL["n_bootstrap"]), 1000)

    df = pd.read_parquet(path)
    test = df.loc[df["split_role"] == "test"].copy() if "split_role" in df.columns else df
    cr = build_cr_frame(test)
    log(f"  Test n={len(cr):,}  defaults={int((cr.cr_event==1).sum()):,}  "
        f"early={int((cr.cr_event==2).sum()):,}  censor={int((cr.cr_event==0).sum()):,}")

    overall = cif_bias(
        cr["cr_time"].to_numpy(),
        cr["cr_event"].to_numpy(),
        horizon_months=horizon_m,
        n_bootstrap=B,
        seed=int(cfg.RANDOM_SEED),
    )
    strata = cif_bias_by_rating(
        test,
        horizon_months=horizon_m,
        n_bootstrap=min(200, B),
        seed=1,
    )

    delta = float(overall["delta"])
    naive_v = float(overall["naive"]["value"])
    aj_v = float(overall["aj"]["value"])
    boot = overall["bootstrap"]
    direction_ok = bool(overall.get("supports_h3_direction"))
    strata_ok = bool(strata.get("h3_strata_rule_met"))
    h3_reject = bool(direction_ok and strata_ok)

    log(
        f"  @ {horizon_m:.0f}m: naive={naive_v:.4f}  AJ={aj_v:.4f}  "
        f"Δ={delta:.4f}  CI=[{boot['ci_low']:.4f},{boot['ci_high']:.4f}]"
    )
    log(
        f"  Strata supporting H3: {strata.get('n_strata_supporting_h3')}/"
        f"{strata.get('n_strata')} (need ≥{min_strata})  "
        f"H3_preview_reject={h3_reject}"
    )

    probe_dir = cfg.DIRS["probes"]
    probe_dir.mkdir(parents=True, exist_ok=True)
    stem = probe_dir / "F01_h3_competing_risks"

    caption = (
        f"On the Bondora 2020 temporal test set, treating early repayment as "
        f"right-censoring (naive) yields a 12-month default CIF of {naive_v:.3f}, "
        f"versus {aj_v:.3f} under Aalen–Johansen with early repayment as a competing "
        f"event (Δ={delta:.3f}; 95% bootstrap CI "
        f"[{boot['ci_low']:.3f}, {boot['ci_high']:.3f}]). "
        f"The positive bias (naive > AJ) is the predicted direction of competing-risks "
        f"blindness; {strata.get('n_strata_supporting_h3')} of "
        f"{strata.get('n_strata')} rating strata meet the H3 stratum rule "
        f"(≥{min_strata} required)."
    )

    doc = {
        "stage": "F01",
        "probe": "h3_competing_risks_bias_bondora",
        "hypothesis": "H3",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "eval_split": "test",
        "n_test": int(len(cr)),
        "horizon_months": horizon_m,
        "delta_threshold": delta_thr,
        "min_strata_consistent": min_strata,
        "overall": overall,
        "by_rating": strata,
        "h3_preview_reject": h3_reject,
        "why_relevant": [
            "Published Bondora baselines treat prepayment-like exits as censoring, "
            "so default risk is estimated as if those loans could still default later.",
            "Competing-risks theory guarantees naive CIF ≥ AJ when the competitor has "
            "positive hazard; H3 asks whether the gap is large enough to matter (≥2pp).",
            "A +2pp shift in 12-month default incidence moves capital / pricing under "
            "credit risk rules — the C-index alone does not show this bias.",
            "Links the anchor thesis (assumption-aligned evaluation) to structured "
            "finance: wrong censoring assumption → wrong absolute risk.",
        ],
        "caption_draft": caption,
        "artifacts_related": [
            "results/ladder/d02_competing_risks.json",
        ],
    }
    write_json(Path(str(stem) + ".json"), doc)

    # Markdown exhibit
    md = [
        "# F01 — H3 Bondora competing-risks bias",
        "",
        f"_Generated UTC: `{doc['generated_at_utc']}`_",
        "",
        "## Result",
        "",
        f"- Horizon: **{horizon_m:.0f} months**",
        f"- Naive CIF (early = censor): **{naive_v:.4f}**",
        f"- AJ CIF (early = competitor): **{aj_v:.4f}**",
        f"- **Δ (naive − AJ)** = **{delta:.4f}** "
        f"(threshold {delta_thr}; CI [{boot['ci_low']:.4f}, {boot['ci_high']:.4f}])",
        f"- Direction naive > AJ: `{direction_ok}`",
        f"- Strata rule (≥{min_strata} with H3 support): "
        f"`{strata.get('n_strata_supporting_h3')}/{strata.get('n_strata')}` → `{strata_ok}`",
        f"- **H3 preview reject:** `{h3_reject}`",
        "",
        "## By rating",
        "",
        "| Rating | n | Δ | CI low | CI high | Supports H3 |",
        "|--------|---|---|--------|---------|-------------|",
    ]
    for s in strata.get("strata") or []:
        b = s.get("bootstrap") or {}
        md.append(
            f"| {s.get('rating')} | {s.get('n')} | {s.get('delta'):.4f} | "
            f"{b.get('ci_low'):.4f} | {b.get('ci_high'):.4f} | "
            f"{s.get('supports_h3_direction')} |"
        )
    md += [
        "",
        "## Why this matters",
        "",
    ]
    for bullet in doc["why_relevant"]:
        md.append(f"- {bullet}")
    md += [
        "",
        "## Caption draft",
        "",
        caption,
        "",
    ]
    Path(str(stem) + ".md").write_text("\n".join(md), encoding="utf-8")

    tex = [
        "% Auto-generated by F01_competing_risks_bias_bondora.py",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Rating & $n$ & $\\Delta$ (naive$-$AJ) & CI low & CI high \\\\",
        "\\midrule",
    ]
    for s in strata.get("strata") or []:
        b = s.get("bootstrap") or {}
        tex.append(
            f"{s.get('rating')} & {s.get('n')} & {s.get('delta'):.4f} & "
            f"{b.get('ci_low'):.4f} & {b.get('ci_high'):.4f} \\\\"
        )
    tex += [
        "\\midrule",
        f"Overall & {len(cr)} & {delta:.4f} & {boot['ci_low']:.4f} & {boot['ci_high']:.4f} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ]
    Path(str(stem) + ".tex").write_text("\n".join(tex), encoding="utf-8")

    with Path(str(stem) + ".csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["rating", "n", "delta", "ci_low", "ci_high", "supports_h3_direction"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in strata.get("strata") or []:
            b = s.get("bootstrap") or {}
            w.writerow(
                {
                    "rating": s.get("rating"),
                    "n": s.get("n"),
                    "delta": s.get("delta"),
                    "ci_low": b.get("ci_low"),
                    "ci_high": b.get("ci_high"),
                    "supports_h3_direction": s.get("supports_h3_direction"),
                }
            )
        w.writerow(
            {
                "rating": "OVERALL",
                "n": len(cr),
                "delta": delta,
                "ci_low": boot["ci_low"],
                "ci_high": boot["ci_high"],
                "supports_h3_direction": direction_ok,
            }
        )

    log(f"  Wrote {stem.relative_to(cfg.ROOT)}.*")
    log("F01 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
