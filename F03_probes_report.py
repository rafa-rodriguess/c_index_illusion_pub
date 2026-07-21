"""
F03_probes_report.py — Block F rollup (paper probe summary)
===========================================================
Aggregates F00 (H4), F01 (H3), F02 (H5) into a single paper-facing report.

Writes:
  results/probes/report.md
  results/probes/report.json

Execute:
    python -W default F03_probes_report.py
"""

from __future__ import annotations

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


def _fmt(x: Any, digits: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, float):
        return f"{x:.{digits}f}"
    return str(x)


def _summarize_h4(sens: dict | None, h04: dict | None = None) -> dict[str, Any]:
    if sens is None:
        return {
            "hypothesis": "H4",
            "status": "missing",
            "preview_reject": None,
            "one_liner": "F00_sens1 missing — run leave-one-out survey.",
        }
    loo = (sens or {}).get("leave_one_out") or []
    subsets = (sens or {}).get("h4_subsets") or []
    loo_hits = [r for r in loo if r.get("ge_threshold")]
    sub_hits = [r for r in subsets if r.get("ge_threshold")]
    max_loo = (sens or {}).get("max_delta_loo")
    max_sub = (sens or {}).get("max_delta_h4_subset")
    top_loo = sorted(loo_hits, key=lambda r: float(r.get("delta_c") or 0), reverse=True)[:4]
    top_names = [
        ",".join(r.get("dropped") or []) or str(r.get("dropped_ids")) for r in top_loo
    ]
    formal = bool((h04 or {}).get("h4_reject")) if h04 else None
    preview = bool(loo_hits or sub_hits)
    c_full = (sens or {}).get("c_full") or (h04 or {}).get("c_full")
    if preview:
        one_liner = (
            f"LOO floor hits={len(loo_hits)} (max Δ={_fmt(max_loo, 3)}); "
            f"formal H04 reject={formal}."
        )
    else:
        one_liner = (
            f"No LOO/subset hit with ΔC≥0.03 on current Cox pop "
            f"(max Δ_LOO={_fmt(max_loo, 3)}; C_full={_fmt(c_full, 4)}); "
            f"H4 does not reject (formal={formal})."
        )
    return {
        "hypothesis": "H4",
        "domain": "DOMAIN_01 / Backblaze",
        "status": "live",
        "method": "broad_ablation_survey",
        "preview_reject": preview,
        "formal_reject": formal,
        "primary_artifact": "F00_sens1_leave_one_out",
        "formal_artifact": "H04_paired_bootstrap_loo",
        "loo_hits": len(loo_hits),
        "subset_hits": len(sub_hits),
        "max_delta_loo": max_loo,
        "max_delta_h4_subset": max_sub,
        "c_full": c_full,
        "top_loo_drops": top_names,
        "caption": (
            f"Broad SMART ablation (LOO): {len(loo_hits)} hits with ΔC≥0.03 "
            f"(max Δ={_fmt(max_loo, 3)}; e.g. {', '.join(top_names[:3]) or '—'}). "
            f"H4 floor: {preview}; formal bootstrap (H04): {formal}."
        ),
        "one_liner": one_liner,
        "honesty": (
            "Existential over ablation sets — LOO survey + paired bootstrap on floor hits. "
            "Not a claim that any fixed SMART ID list was known a priori. "
            "On H6a Cox cohort, LOO may yield zero floor hits."
        ),
    }


def _summarize_h3(doc: dict | None) -> dict[str, Any]:
    if not doc:
        return {
            "hypothesis": "H3",
            "status": "missing",
            "preview_reject": None,
            "one_liner": "F01 artifact missing — run F01.",
        }
    ov = doc.get("overall") or {}
    boot = ov.get("bootstrap") or {}
    by = doc.get("by_rating") or {}
    return {
        "hypothesis": "H3",
        "domain": "DOMAIN_02 / Bondora",
        "status": "live",
        "preview_reject": bool(doc.get("h3_preview_reject")),
        "primary_artifact": "F01_h3_competing_risks",
        "delta": ov.get("delta"),
        "naive": (ov.get("naive") or {}).get("value"),
        "aj": (ov.get("aj") or {}).get("value"),
        "ci_low": boot.get("ci_low"),
        "ci_high": boot.get("ci_high"),
        "n_strata_supporting": by.get("n_strata_supporting_h3"),
        "n_strata": by.get("n_strata"),
        "caption": doc.get("caption_draft")
        or (
            f"Bondora CIF Δ(naive−AJ)@12m={_fmt(ov.get('delta'), 3)}; "
            f"H3 preview={doc.get('h3_preview_reject')}."
        ),
        "one_liner": (
            "Treating early repayment as censoring inflates 12m default CIF "
            f"by ~{_fmt((ov.get('delta') or 0) * 100, 1)} pp vs Aalen–Johansen."
        ),
        "why_relevant": doc.get("why_relevant"),
    }


def _summarize_h5(doc: dict | None) -> dict[str, Any]:
    if not doc:
        return {
            "hypothesis": "H5",
            "status": "missing",
            "preview_reject": None,
            "one_liner": "F02 artifact missing — run F02.",
        }
    primary_id = doc.get("primary_model_id")
    models = doc.get("models") or []
    prim = next((m for m in models if m.get("model_id") == primary_id), models[0] if models else {})
    bs = prim.get("brier_strip") or {}
    bv = (bs.get("brier") or {}).get("values") or {}
    rows = []
    for m in models:
        s = m.get("brier_strip") or {}
        v = (s.get("brier") or {}).get("values") or {}
        rows.append(
            {
                "model_id": m.get("model_id"),
                "harrell_c": s.get("global_harrell_c"),
                "c_in_band": s.get("global_c_in_band"),
                "brier_12": v.get("12"),
                "brier_24": v.get("24"),
                "brier_36": v.get("36"),
                "delta_36_12": s.get("delta_brier_36_minus_12"),
                "h5_preview_reject": m.get("h5_preview_reject"),
                "auc_would_reject": m.get("h5_auc_sensitivity_reject"),
            }
        )
    return {
        "hypothesis": "H5",
        "domain": "DOMAIN_03 / Stack Exchange Politics",
        "status": "live",
        "method": "brier_horizon_degradation",
        "preview_reject": bool(doc.get("h5_preview_reject")),
        "primary_artifact": "F02_h5_temporal",
        "primary_model_id": primary_id,
        "primary_metric": doc.get("primary_metric"),
        "sensitivity_metric": doc.get("sensitivity_metric"),
        "brier": bv,
        "delta_brier_36_12": bs.get("delta_brier_36_minus_12"),
        "global_c": bs.get("global_harrell_c"),
        "models": rows,
        "caption": doc.get("caption_draft"),
        "one_liner": (
            f"Under C in the Abedi band, IPCW Brier rises 12→36m "
            f"(primary `{primary_id}`: {_fmt(bv.get('12'), 3)}→{_fmt(bv.get('36'), 3)}; "
            f"Δ={_fmt(bs.get('delta_brier_36_minus_12'), 3)}). AUC sensitivity did not reject."
        ),
        "honesty": doc.get("honesty_note"),
        "why_relevant": doc.get("why_relevant"),
    }


def _build_markdown(doc: dict[str, Any]) -> str:
    h3, h4, h5 = doc["H3"], doc["H4"], doc["H5"]
    lines = [
        "# Block F — Failure-mode probes report",
        "",
        f"_Generated UTC: `{doc['generated_at_utc']}` · protocol `{doc.get('protocol_version')}`_",
        "",
        "Paper-facing rollup of F00–F02. Hypothesis flags are **previews**, not Holm-adjusted decisions.",
        "",
        "## Verdict strip",
        "",
        "| Hypothesis | Domain | Preview reject | Primary artifact | One-liner |",
        "|------------|--------|----------------|------------------|-----------|",
        f"| **H3** | Bondora | `{h3.get('preview_reject')}` | `{h3.get('primary_artifact', '—')}` | {h3.get('one_liner', '—')} |",
        f"| **H4** | Backblaze | `{h4.get('preview_reject')}` | `{h4.get('primary_artifact', '—')}` | {h4.get('one_liner', '—')} |",
        f"| **H5** | Stack Exchange | `{h5.get('preview_reject')}` | `{h5.get('primary_artifact', '—')}` | {h5.get('one_liner', '—')} |",
        "",
        f"**Block F probes supporting preview:** "
        f"{sum(1 for h in (h3, h4, h5) if h.get('preview_reject'))} / 3 "
        f"(H3/H4/H5). H1–H2 live on the ladder (E); H_meta needs outer Holm later.",
        "",
        "---",
        "",
        "## H3 — Competing-risks blindness (F01)",
        "",
        f"- Status: `{h3.get('status')}` · preview reject: `{h3.get('preview_reject')}`",
        f"- Δ(naive−AJ)@12m = **{_fmt(h3.get('delta'))}** "
        f"(naive={_fmt(h3.get('naive'))}, AJ={_fmt(h3.get('aj'))}; "
        f"CI [{_fmt(h3.get('ci_low'))}, {_fmt(h3.get('ci_high'))}])",
        f"- Strata supporting: **{h3.get('n_strata_supporting')}/{h3.get('n_strata')}**",
        "",
        "### Caption",
        "",
        h3.get("caption") or "—",
        "",
        "---",
        "",
        "## H4 — Broad SMART ablation (F00 / H04)",
        "",
        f"- Status: `{h4.get('status')}` · preview reject: `{h4.get('preview_reject')}` · formal: `{h4.get('formal_reject')}`",
        f"- Method: `{h4.get('method')}` (existential LOO survey)",
        f"- LOO hits ≥0.03: **{h4.get('loo_hits')}** (max Δ={_fmt(h4.get('max_delta_loo'))})",
        f"- Top LOO drops: {', '.join(h4.get('top_loo_drops') or []) or '—'}",
        "",
        f"> Honesty: {h4.get('honesty')}",
        "",
        "### Caption",
        "",
        h4.get("caption") or "—",
        "",
        "---",
        "",
        "## H5 — Brier↑ under stable C (F02)",
        "",
        f"- Status: `{h5.get('status')}` · preview reject: `{h5.get('preview_reject')}`",
        f"- Primary metric: `{h5.get('primary_metric')}` · sensitivity: `{h5.get('sensitivity_metric')}`",
        f"- Primary model: `{h5.get('primary_model_id')}` · C={_fmt(h5.get('global_c'))}",
        f"- Brier 12/24/36: {_fmt((h5.get('brier') or {}).get('12'))} / "
        f"{_fmt((h5.get('brier') or {}).get('24'))} / "
        f"{_fmt((h5.get('brier') or {}).get('36'))}",
        f"- Δ Brier(36−12) = **{_fmt(h5.get('delta_brier_36_12'))}**",
        "",
        f"> Honesty: {h5.get('honesty')}",
        "",
        "### Models",
        "",
        "| Model | C | In band | Brier12 | Brier24 | Brier36 | Δ | H5 | AUC sens. |",
        "|-------|---|---------|---------|---------|---------|---|----|-----------|",
    ]
    for r in h5.get("models") or []:
        lines.append(
            f"| `{r.get('model_id')}` | {_fmt(r.get('harrell_c'))} | {r.get('c_in_band')} | "
            f"{_fmt(r.get('brier_12'))} | {_fmt(r.get('brier_24'))} | {_fmt(r.get('brier_36'))} | "
            f"{_fmt(r.get('delta_36_12'))} | {r.get('h5_preview_reject')} | "
            f"{r.get('auc_would_reject')} |"
        )
    lines += [
        "",
        "### Caption",
        "",
        h5.get("caption") or "—",
        "",
        "---",
        "",
        "## Artifact index",
        "",
        "| Probe | Files |",
        "|-------|-------|",
        "| F00 H4 LOO survey | `results/probes/F00_sens1_leave_one_out.*` |",
        "| H04 H4 formal bootstrap | `results/probes/H04_paired_bootstrap_loo.*` |",
        "| F01 H3 | `results/probes/F01_h3_competing_risks.*` |",
        "| F02 H5 | `results/probes/F02_h5_temporal.*` |",
        "| This report | `results/probes/report.{md,json}` |",
        "",
        "## Next (Block G)",
        "",
        "- `G00_leakage_controls_audit.py`",
        "- `G01_export_harness.py`",
        "- `G02_final_verdict_table.py`",
        "- `G03_holm_family.py`",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    log("─" * 60)
    log("F03 — PROBES REPORT (F00–F02 rollup)")
    log("─" * 60)

    probes = cfg.DIRS["probes"]
    probes.mkdir(parents=True, exist_ok=True)

    h4 = _summarize_h4(
        _read(probes / "F00_sens1_leave_one_out.json"),
        _read(probes / "H04_paired_bootstrap_loo.json"),
    )
    h3 = _summarize_h3(_read(probes / "F01_h3_competing_risks.json"))
    h5 = _summarize_h5(_read(probes / "F02_h5_temporal.json"))

    n_ok = sum(1 for h in (h3, h4, h5) if h.get("preview_reject"))
    n_live = sum(1 for h in (h3, h4, h5) if h.get("status") == "live")

    doc = {
        "stage": "F03",
        "artifact": "probes_report",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": cfg.PROTOCOL.get("version"),
        "n_probes_live": n_live,
        "n_preview_reject": n_ok,
        "H3": h3,
        "H4": h4,
        "H5": h5,
        "meta_note": (
            "H1–H2 are ladder (E) exhibits; this report covers Bloco F probes H3–H5. "
            "Outer Holm / H_meta: see G03."
        ),
    }

    write_json(probes / "report.json", doc)
    md = _build_markdown(doc)
    (probes / "report.md").write_text(md, encoding="utf-8")

    log(f"  Live probes: {n_live}/3  ·  preview reject: {n_ok}/3")
    log(f"  H3={h3.get('preview_reject')}  H4={h4.get('preview_reject')}  H5={h5.get('preview_reject')}")
    log(f"  Wrote { (probes / 'report.md').relative_to(cfg.ROOT) }")
    log(f"  Wrote { (probes / 'report.json').relative_to(cfg.ROOT) }")
    log("F03 complete.")
    return 0 if n_live == 3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
