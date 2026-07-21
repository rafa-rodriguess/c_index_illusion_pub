"""
E06_ladder_summary.py — Cross-domain ladder rollup
==================================================
Merges E01–E05 into:
  - ``results/ladder/summary.{json,md}`` (ops dashboard)
  - ``results/reproduction/LADDER_evaluation_table.{json,csv,md,tex}``
    (paper-ready evaluation-ladder exhibit; partial until D3 + F probes)

Hypothesis flags here are **previews** (not Holm-final decisions).

Execute:
    python -W default E06_ladder_summary.py
"""

from __future__ import annotations

import csv
import json
import sys
import warnings
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import cfg
from src.metrics.freeze import load_frozen_manifest
from src.metrics.io import utc_now, write_json

warnings.filterwarnings("default")

DOMAIN_LABEL = {
    "DOMAIN_01": "Backblaze (D1)",
    "DOMAIN_02": "Bondora (D2)",
    "DOMAIN_03": "Stack Exchange (D3)",
}


def log(msg: str = "") -> None:
    print(msg, flush=True)


def _read(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(x: Any, nd: int = 4, *, sci_lt: float | None = None) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "yes" if x else "no"
    try:
        v = float(x)
        if sci_lt is not None and 0 < abs(v) < sci_lt:
            return f"{v:.1e}"
        return f"{v:.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def _collect_model_rows(ladder_dir: Path) -> list[dict[str, Any]]:
    """One row per domain×model with metrics pulled from live rungs."""
    rows: list[dict[str, Any]] = []
    for domain_id, label in DOMAIN_LABEL.items():
        slug = domain_id.lower().replace("domain_", "d")
        disc = _read(ladder_dir / f"{slug}_discrimination.json") or {}
        cal = _read(ladder_dir / f"{slug}_calibration.json") or {}
        sco = _read(ladder_dir / f"{slug}_scores.json") or {}
        dep = _read(ladder_dir / f"{slug}_dependent_censoring.json") or {}
        cr = _read(ladder_dir / f"{slug}_competing_risks.json") or {}

        cal_by = {m.get("model_id"): m for m in cal.get("models") or []}
        sco_by = {m.get("model_id"): m for m in sco.get("models") or []}
        dep_by = {m.get("model_id"): m for m in dep.get("models") or []}

        models = disc.get("models") or []
        if not models:
            # still emit a placeholder for domains without discrimination models
            rows.append(
                {
                    "domain_id": domain_id,
                    "domain_label": label,
                    "model_id": "—",
                    "harrell_c": None,
                    "uno_c": None,
                    "dcal_p": None,
                    "dcal_reject": None,
                    "ipcw_ibs": None,
                    "cg_max_mae_vs_km": None,
                    "cif_delta_12m": None,
                    "status_note": cr.get("implementation_status")
                    or disc.get("implementation_status")
                    or "missing",
                }
            )
            continue

        for m in models:
            mid = m.get("model_id")
            cm = cal_by.get(mid) or {}
            sm = sco_by.get(mid) or {}
            dm = dep_by.get(mid) or {}
            dcal = cm.get("d_calibration") or {}
            ibs = sm.get("ipcw_ibs") or {}
            sweep = dm.get("sweep") or {}

            # Domain-level CIF once (on classical Cox row)
            cif_delta = None
            if (
                domain_id == "DOMAIN_02"
                and cr.get("implementation_status") == "live"
                and mid == "cox_classical"
            ):
                cif_delta = (cr.get("overall_test") or {}).get("delta")

            st_parts = []
            for name, blob in (
                ("E01", m),
                ("E02", cm),
                ("E03", sm),
                ("E04", dm),
            ):
                st = blob.get("implementation_status")
                if st and st not in {"live", "live_population_only"}:
                    st_parts.append(f"{name}:{st}")
            if domain_id != "DOMAIN_02" and cr.get("implementation_status") == "not_applicable":
                st_parts.append("E05:N/A")

            rows.append(
                {
                    "domain_id": domain_id,
                    "domain_label": label,
                    "model_id": mid,
                    "h1_rank_key": m.get("h1_rank_key"),
                    "harrell_c": (m.get("harrell") or {}).get("value")
                    if m.get("implementation_status") == "live"
                    else None,
                    "uno_c": (m.get("uno") or {}).get("value")
                    if m.get("implementation_status") == "live"
                    else None,
                    "dcal_p": dcal.get("p_value")
                    if cm.get("implementation_status") == "live"
                    else None,
                    "dcal_reject": dcal.get("reject_h0_well_calibrated")
                    if cm.get("implementation_status") == "live"
                    else None,
                    "ipcw_ibs": ibs.get("value")
                    if sm.get("implementation_status") == "live"
                    else None,
                    "cg_max_mae_vs_km": sweep.get("max_mae_cg_vs_km")
                    if str(dm.get("implementation_status", "")).startswith("live")
                    else None,
                    "cif_delta_12m": cif_delta,
                    "status_note": "; ".join(st_parts) if st_parts else "live",
                }
            )
    return rows


def _hypothesis_flags(rows: list[dict[str, Any]], ladder_dir: Path) -> dict[str, Any]:
    # H2: exists model with C_H >= 0.90 and D-Cal reject
    h2_hits = []
    for r in rows:
        c = r.get("harrell_c")
        rej = r.get("dcal_reject")
        if c is not None and rej is True and float(c) >= 0.90:
            h2_hits.append(f"{r['domain_id']}/{r['model_id']}")

    # H1: need ≥2 models in a domain with both C and IBS to rank
    by_dom: dict[str, list] = {}
    for r in rows:
        if r.get("harrell_c") is not None and r.get("ipcw_ibs") is not None:
            by_dom.setdefault(r["domain_id"], []).append(r)
    h1_ready_domains = [d for d, ms in by_dom.items() if len(ms) >= 2]
    h1_status = (
        "preview_possible"
        if h1_ready_domains
        else "incomplete_need_ge2_scored_models_per_domain"
    )

    # H3 from F01 probe when present; else E05 ladder artifact
    f01 = _read(cfg.DIRS["probes"] / "F01_h3_competing_risks.json") or {}
    cr = _read(ladder_dir / "d02_competing_risks.json") or {}
    if f01.get("overall"):
        ov = f01["overall"]
        boot = ov.get("bootstrap") or {}
        h3_reject = bool(f01.get("h3_preview_reject"))
        h3_status = (
            "preview_supported" if h3_reject else "preview_not_supported"
        )
        h3 = {
            "h3_decision_preview": h3_reject,
            "delta": ov.get("delta"),
            "naive": (ov.get("naive") or {}).get("value"),
            "aj": (ov.get("aj") or {}).get("value"),
            "ci_low": boot.get("ci_low"),
            "ci_high": boot.get("ci_high"),
            "n_strata_supporting": (f01.get("by_rating") or {}).get(
                "n_strata_supporting_h3"
            ),
            "source": "F01",
        }
    else:
        h3 = (cr.get("h3_preview") or {}) if cr.get("implementation_status") == "live" else {}
        h3_status = (
            "preview_supported"
            if h3.get("h3_decision_preview")
            else (
                "preview_not_supported"
                if cr.get("implementation_status") == "live"
                else "pending_F01"
            )
        )

    # H4 from F00 LOO survey (+ H04 formal close when present)
    f00_sens = _read(cfg.DIRS["probes"] / "F00_sens1_leave_one_out.json") or {}
    h04 = _read(cfg.DIRS["probes"] / "H04_paired_bootstrap_loo.json") or {}
    if f00_sens.get("leave_one_out") is not None:
        loo = f00_sens.get("leave_one_out") or []
        subsets = f00_sens.get("h4_subsets") or []
        if not subsets:
            for k, v in f00_sens.items():
                if k.startswith("h4") and isinstance(v, list):
                    subsets = v
                    break
        loo_hits = [r for r in loo if r.get("ge_threshold")]
        sub_hits = [r for r in subsets if r.get("ge_threshold")]
        any_hit = bool(loo_hits or sub_hits)
        max_loo = f00_sens.get("max_delta_loo")
        max_sub = f00_sens.get("max_delta_h4_subset")
        formal = h04.get("h4_reject")
        h4 = {
            "status": "preview_supported" if any_hit else "preview_not_supported",
            "method": "broad_ablation_survey",
            "loo_hits": len(loo_hits),
            "subset_hits": len(sub_hits),
            "max_delta_loo": max_loo,
            "max_delta_h4_subset": max_sub,
            "formal_reject": formal,
            "note": (
                f"F00 LOO survey: hits={len(loo_hits)} (max Δ={max_loo}); "
                f"subset hits={len(sub_hits)} (max Δ={max_sub}). "
                f"H04 formal reject={formal}."
            ),
        }
    elif h04.get("h4_reject") is not None:
        h4 = {
            "status": "preview_supported" if h04.get("h4_reject") else "preview_not_supported",
            "formal_reject": h04.get("h4_reject"),
            "note": f"H04 only: h4_reject={h04.get('h4_reject')}",
        }
    else:
        h4 = {
            "status": "pending_F00_ablation",
            "note": "Requires F00 leave-one-out survey on D1.",
        }

    h5_flag = _h5_from_f02()

    return {
        "H1": {
            "status": h1_status,
            "ready_domains": h1_ready_domains,
            "note": "Formal τ_K + permutation deferred; need XGB curves or ablated Cox.",
        },
        "H2": {
            "status": "preview_supported" if h2_hits else "preview_not_supported",
            "hits": h2_hits,
            "note": "C_H>=0.90 and D-Calibration reject (SurvivalEVAL).",
        },
        "H3": {
            "status": h3_status,
            "preview": h3,
            "note": (
                f"F01: Δ={h3.get('delta')} "
                f"(naive={h3.get('naive')}, AJ={h3.get('aj')}); "
                f"CI=[{h3.get('ci_low')}, {h3.get('ci_high')}]; "
                f"strata={h3.get('n_strata_supporting')}."
                if h3.get("source") == "F01"
                else "CIF naive−AJ @12m on Bondora test (E05); prefer F01 probe."
            ),
        },
        "H4": h4,
        "H5": h5_flag,
    }


def _h5_from_f02() -> dict[str, Any]:
    f02 = _read(cfg.DIRS["probes"] / "F02_h5_temporal.json") or {}
    if not f02.get("models"):
        return {
            "status": "pending_F02_temporal_probe",
            "note": (
                "D3 RSF is on the ladder (E01–E04); H5 still needs F02 "
                "horizon Brier strip on θ∈{12,24,36}."
            ),
        }
    any_reject = bool(f02.get("h5_preview_reject"))
    primary = f02.get("primary_model_id")
    models = f02.get("models") or []
    prim = next((m for m in models if m.get("model_id") == primary), models[0])
    # Prefer Brier strip (new primary); fall back to legacy auc_strip key
    strip = prim.get("brier_strip") or prim.get("auc_strip") or {}
    brier_vals = (strip.get("brier") or {}).get("values") or {}
    auc_sens = prim.get("auc_strip_sensitivity") or prim.get("auc_strip") or {}
    auc_vals = (auc_sens.get("auc") or {}).get("values") or {}
    return {
        "status": "preview_supported" if any_reject else "preview_not_supported",
        "primary_model_id": primary,
        "primary_metric": f02.get("primary_metric") or "ipcw_brier_at_horizons",
        "delta_brier_36_12": strip.get("delta_brier_36_minus_12"),
        "brier": brier_vals,
        "auc_sensitivity": auc_vals,
        "global_c": strip.get("global_harrell_c"),
        "note": (
            f"F02: primary={primary} C={strip.get('global_harrell_c')} "
            f"Brier(12/24/36)={brier_vals.get('12')}/{brier_vals.get('24')}/{brier_vals.get('36')} "
            f"ΔB={strip.get('delta_brier_36_minus_12')}; "
            f"reject_any={any_reject}."
        ),
    }


def _paper_markdown(rows: list[dict[str, Any]], flags: dict[str, Any], meta: dict) -> str:
    lines = [
        "# Evaluation ladder (Block E) — paper exhibit",
        "",
        f"_Generated UTC: `{meta['generated_at_utc']}` · protocol `{meta.get('protocol_version')}`_",
        "",
        "> Partial until D3 is on the ladder and Bloco F probes (H4/H5) run. "
        "Hypothesis flags below are **previews**, not Holm-adjusted decisions.",
        "",
        "## Ladder metrics by domain / frozen model",
        "",
        "| Domain | Model | Harrell C | Uno C | D-Cal p | Reject | IPCW-IBS | "
        "CG max MAE | CIF Δ@12m | Notes |",
        "|--------|-------|-----------|-------|---------|--------|----------|"
        "-----------|-----------|-------|",
    ]
    for r in rows:
        lines.append(
            "| {domain} | `{model}` | {c} | {u} | {p} | {rej} | {ibs} | {cg} | {cif} | {note} |".format(
                domain=r["domain_label"],
                model=r["model_id"],
                c=_fmt(r["harrell_c"]),
                u=_fmt(r["uno_c"]),
                p=_fmt(r["dcal_p"], sci_lt=1e-3),
                rej=_fmt(r["dcal_reject"]),
                ibs=_fmt(r["ipcw_ibs"]),
                cg=_fmt(r["cg_max_mae_vs_km"]),
                cif=_fmt(r["cif_delta_12m"]),
                note=r.get("status_note") or "—",
            )
        )

    lines += [
        "",
        "## Hypothesis previews",
        "",
        f"- **H1** (C vs IBS ranking): `{flags['H1']['status']}`",
        f"- **H2** (high C fails D-Cal): `{flags['H2']['status']}`"
        + (f" — hits: {', '.join(flags['H2']['hits'])}" if flags["H2"].get("hits") else ""),
        f"- **H3** (CR blindness Bondora): `{flags['H3']['status']}`",
        f"- **H4** (Backblaze broad SMART ablation): `{flags['H4']['status']}`",
        f"- **H5** (Stack Exchange Brier horizon): `{flags['H5']['status']}`",
        "",
        "## Caption draft (Results / Evaluation)",
        "",
        "_Assumption-aligned evaluation ladder applied to frozen reproductions of "
        "published non-clinical survival models (no retrain). Discrimination and "
        "IPCW-IBS use sksurv / SurvivalEVAL; D-Calibration uses SurvivalEVAL; "
        "competing-risk CIF uses Aalen–Johansen (lifelines) with early repayment "
        "as the Bondora competitor. Domain 3 pending estimator export._",
        "",
    ]
    return "\n".join(lines)


def _paper_tex(rows: list[dict[str, Any]]) -> str:
    lines = [
        "% Auto-generated by E06_ladder_summary.py — evaluation ladder exhibit",
        "\\begin{tabular}{llrrrrrrl}",
        "\\toprule",
        "Domain & Model & Harrell $C$ & Uno $C$ & D-Cal $p$ & "
        "IPCW-IBS & CG MAE & CIF $\\Delta_{12m}$ \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(
            "{domain} & {model} & {c} & {u} & {p} & {ibs} & {cg} & {cif} \\\\".format(
                domain=r["domain_label"].replace("&", "\\&"),
                model=str(r["model_id"]).replace("_", "\\_"),
                c=_fmt(r["harrell_c"]),
                u=_fmt(r["uno_c"]),
                p=_fmt(r["dcal_p"], sci_lt=1e-3),
                ibs=_fmt(r["ipcw_ibs"]),
                cg=_fmt(r["cg_max_mae_vs_km"]),
                cif=_fmt(r["cif_delta_12m"]),
            )
        )
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def _write_paper_assets(
    rows: list[dict[str, Any]],
    flags: dict[str, Any],
    meta: dict[str, Any],
) -> list[Path]:
    out_dir = cfg.DIRS["reproduction"]
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / "LADDER_evaluation_table"

    doc = {
        "artifact": "ladder_evaluation_table",
        "schema_version": "2026-07-11.ladder_table.v1",
        "generated_at_utc": meta["generated_at_utc"],
        "protocol_version": meta.get("protocol_version"),
        "partial": True,
        "partial_reasons": [
            "DOMAIN_03 not on prediction ladder",
            "H1 incomplete without ≥2 curve-scored models/domain",
            "H4/H5 require Bloco F",
            "Flags are previews (no outer Holm yet)",
        ],
        "hypothesis_flags": flags,
        "rows": rows,
        "caption_draft": (
            "Assumption-aligned evaluation ladder on frozen domain reproductions "
            "(Block E). Metrics via SurvivalEVAL / sksurv / lifelines AJ; no retrain."
        ),
    }
    paths = []
    jp = Path(str(stem) + ".json")
    write_json(jp, doc)
    paths.append(jp)

    mp = Path(str(stem) + ".md")
    mp.write_text(_paper_markdown(rows, flags, meta), encoding="utf-8")
    paths.append(mp)

    tp = Path(str(stem) + ".tex")
    tp.write_text(_paper_tex(rows), encoding="utf-8")
    paths.append(tp)

    cp = Path(str(stem) + ".csv")
    cols = [
        "domain_id",
        "domain_label",
        "model_id",
        "harrell_c",
        "uno_c",
        "dcal_p",
        "dcal_reject",
        "ipcw_ibs",
        "cg_max_mae_vs_km",
        "cif_delta_12m",
        "status_note",
    ]
    with cp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    paths.append(cp)
    return paths


def _ops_markdown(summary: dict) -> str:
    lines = [
        "# Anchor ladder summary (Block E)",
        "",
        f"- Generated (UTC): `{summary['generated_at_utc']}`",
        f"- Protocol: `{summary.get('protocol_version')}`",
        f"- D99 report: `{summary.get('d99_report')}`",
        f"- Manifest present: "
        f"`{summary['manifest']['n_present']}/{summary['manifest']['n_artifacts']}`",
        f"- Paper table: `{summary.get('paper_table')}`",
        "",
        "## Hypothesis previews",
        "",
    ]
    for hid, blob in (summary.get("hypothesis_flags") or {}).items():
        lines.append(f"- **{hid}**: `{blob.get('status')}` — {blob.get('note', '')}")
    lines += [
        "",
        "## Model-level snapshot",
        "",
        "| Domain | Model | C | D-Cal reject | IBS | CIF Δ |",
        "|--------|-------|---|--------------|-----|-------|",
    ]
    for r in summary.get("model_rows") or []:
        lines.append(
            f"| {r['domain_label']} | `{r['model_id']}` | {_fmt(r['harrell_c'])} | "
            f"{_fmt(r['dcal_reject'])} | {_fmt(r['ipcw_ibs'])} | {_fmt(r['cif_delta_12m'])} |"
        )
    lines += ["", "## Notes", "", summary.get("notes", ""), ""]
    return "\n".join(lines)


def main() -> int:
    log("─" * 60)
    log("E06 — LADDER SUMMARY + PAPER TABLE")
    log("─" * 60)

    manifest = load_frozen_manifest()
    ladder_dir = cfg.DIRS["ladder"]
    rows = _collect_model_rows(ladder_dir)
    flags = _hypothesis_flags(rows, ladder_dir)
    meta = {
        "generated_at_utc": utc_now(),
        "protocol_version": cfg.PROTOCOL.get("version")
        or manifest.get("protocol_version"),
    }
    paper_paths = _write_paper_assets(rows, flags, meta)

    summary = {
        "stage": "E06",
        "artifact": "ladder_summary",
        "generated_at_utc": meta["generated_at_utc"],
        "protocol_version": meta["protocol_version"],
        "d99_report": manifest.get("d99_report"),
        "manifest": {
            "n_artifacts": manifest.get("n_artifacts"),
            "n_present": manifest.get("n_present"),
            "n_predict_ready": manifest.get("n_predict_ready"),
        },
        "model_rows": rows,
        "hypothesis_flags": flags,
        "paper_table": str(paper_paths[0].relative_to(cfg.ROOT)),
        "paper_assets": [str(p.relative_to(cfg.ROOT)) for p in paper_paths],
        "notes": (
            "Live rollup of E01–E05. Paper exhibit is marked partial until D3 "
            "joins the ladder and F00/F02 close H4/H5. Do not treat hypothesis "
            "previews as final Holm decisions."
        ),
    }

    json_path = ladder_dir / "summary.json"
    md_path = ladder_dir / "summary.md"
    write_json(json_path, summary)
    md_path.write_text(_ops_markdown(summary), encoding="utf-8")

    log(f"  Wrote {json_path.relative_to(cfg.ROOT)}")
    log(f"  Wrote {md_path.relative_to(cfg.ROOT)}")
    for p in paper_paths:
        log(f"  Paper: {p.relative_to(cfg.ROOT)}")
    for hid, blob in flags.items():
        log(f"  {hid}: {blob['status']}")
    log("E06 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
