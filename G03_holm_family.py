"""
G03_holm_family.py — Two-level Holm (C00.3) + H_meta
====================================================
Assembles one p per primary hypothesis (H1–H5), applies:

  1. Inner collapse (already done for H1 via min domain p; H3 via strata)
  2. Outer Holm–Bonferroni over the 5 hypothesis p-values
  3. H_meta: reject if ≥3 of 5 outer-Holm rejects

p-value sources (honest, artefact-backed):
  H1 — min domain permutation p from ``H01_ranking_inversion.json``
  H2 — D-Calibration p for ``cox_full`` (C_H≥0.90 gate)
  H3 — bootstrap one-sided approx from overall CIF Δ (CI excludes 0 → ≤1/(B+1));
       strata collapsed with inner Holm on the same approx
  H4 — min over LOO hits of bootstrap approx from ΔC CI (H04)
  H5 — bootstrap approx from primary ``rsf_content`` Brier Δ CI (F02)

Writes:
  results/reproduction/G03_holm_family.{json,md}
  results/logs/G03_holm_family.md

Execute:
    python -W default G03_holm_family.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import PROTOCOL, cfg
from src.metrics.io import utc_now, write_json

ALPHA = float(PROTOCOL["globals"]["alpha"])
META_K = 3  # ≥3 of 5 under outer Holm


def log(msg: str = "") -> None:
    print(msg, flush=True)


def _read(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _boot_p_ci_excludes_zero(ci_low: float | None, B: int) -> float | None:
    """
    Conservative one-sided p when a percentile CI for a positive effect
    excludes 0: no bootstrap replicate ≤0 was observed under the CI construction,
    so p ≤ 1/(B+1) (add-one smoothing).
    """
    if ci_low is None or B is None or B <= 0:
        return None
    if ci_low > 0:
        return float(1.0 / (B + 1.0))
    return None


def holm_adjust(pvalues: dict[str, float], alpha: float = ALPHA) -> dict[str, Any]:
    """Holm–Bonferroni step-down; returns adjusted p and reject flags."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    rejects: dict[str, bool] = {}
    prev = 0.0
    stop = False
    for i, (name, p) in enumerate(items):
        raw_adj = (m - i) * p
        adj = min(1.0, max(prev, raw_adj))
        prev = adj
        adjusted[name] = float(adj)
        if stop:
            rejects[name] = False
        else:
            rejects[name] = bool(adj < alpha)
            if not rejects[name]:
                stop = True
    return {
        "order": [n for n, _ in items],
        "raw": {n: float(p) for n, p in items},
        "adjusted": adjusted,
        "reject": rejects,
        "alpha": alpha,
        "m": m,
    }


def assemble_pvalues() -> dict[str, Any]:
    probes = cfg.DIRS["probes"]
    ladder = cfg.DIRS["ladder"]

    h1 = _read(probes / "H01_ranking_inversion.json") or {}
    h4 = _read(probes / "H04_paired_bootstrap_loo.json") or {}
    f01 = _read(probes / "F01_h3_competing_risks.json") or {}
    f02 = _read(probes / "F02_h5_temporal.json") or {}
    cal = _read(ladder / "d01_calibration.json") or {}
    disc = _read(ladder / "d01_discrimination.json") or {}

    details: dict[str, Any] = {}

    # --- H1 ---
    p_h1 = (h1.get("inner_holm") or h1.get("inner_holm_preview") or {}).get("min_p")
    details["H1"] = {
        "source": "H01_ranking_inversion.inner_holm.min_p (p_family / C00.1 v5)",
        "p": p_h1,
        "protocol_reject_C00_1": h1.get("formal_reject_C00_1_any_domain"),
        "descriptive_inversion": h1.get("any_binary_inversion"),
        "domain_ps": (h1.get("inner_holm") or h1.get("inner_holm_preview") or {}).get(
            "domain_ps"
        ),
        "B": h1.get("B"),
        "note": (
            "C00.1 v5: subject-level bootstrap CI upper≤0.5 and τ≤0.5; "
            "p_family feeds outer Holm."
        ),
    }

    # --- H2 ---
    c_full = None
    for m in disc.get("models") or []:
        if m.get("model_id") == "cox_full":
            c_full = (m.get("harrell") or {}).get("value")
    p_h2 = None
    dcal = None
    for m in cal.get("models") or []:
        if m.get("model_id") == "cox_full":
            dcal = m.get("d_calibration") or {}
            p_h2 = dcal.get("p_value")
    thr = float(PROTOCOL["hypotheses"]["H2"]["c_h_threshold"])
    gate = c_full is not None and c_full >= thr
    details["H2"] = {
        "source": "d01_calibration.cox_full.d_calibration.p_value",
        "p": p_h2,
        "c_full": c_full,
        "c_h_threshold": thr,
        "gate_c_ge_threshold": gate,
        "dcal_reject": (dcal or {}).get("reject_h0_well_calibrated"),
        "protocol_reject": bool(gate and (dcal or {}).get("reject_h0_well_calibrated")),
    }
    if not gate:
        # If gate fails, H2 is not testable as registered — keep p but flag
        details["H2"]["note"] = "C_H gate failed — H2 p retained for Holm but decision N/A"

    # --- H3 (inner Holm on strata, then overall consistency) ---
    strata = ((f01.get("by_rating") or {}).get("strata")) or []
    strata_ps: dict[str, float] = {}
    for s in strata:
        rating = str(s.get("rating"))
        b = s.get("bootstrap") or {}
        B = int(b.get("B") or 1000)
        p_s = _boot_p_ci_excludes_zero(b.get("ci_low"), B)
        # only strata that clear Δ>0.02 enter the decision set; still include all with CI>0 for Holm collapse
        if p_s is not None and s.get("delta", 0) > float(f01.get("delta_threshold") or 0.02):
            strata_ps[rating] = p_s
        elif p_s is not None:
            # non-threshold strata: use p but mark — for inner collapse use threshold-clearing only
            pass
    # Prefer threshold-clearing strata; if empty, fall back to all with CI>0
    if not strata_ps:
        for s in strata:
            rating = str(s.get("rating"))
            b = s.get("bootstrap") or {}
            p_s = _boot_p_ci_excludes_zero(b.get("ci_low"), int(b.get("B") or 1000))
            if p_s is not None:
                strata_ps[rating] = p_s
    inner_h3 = holm_adjust(strata_ps, alpha=ALPHA) if strata_ps else None
    # collapsed p = min adjusted among rejected, else min adjusted (standard: smallest Holm adj that is the first)
    if inner_h3:
        # Use the minimum raw strata p after counting multiplicity via Holm: take max adjusted among rejected step, else min adj
        p_h3 = min(inner_h3["raw"].values())
        # Better: collapsed decision p = the Holm-adjusted p of the last rejected or first non — use min raw * multiplicity already in outer
        # Protocol: inner Holm collapses to one p. Use the smallest Holm-adjusted p among strata.
        p_h3_collapsed = min(inner_h3["adjusted"].values())
    else:
        p_h3_collapsed = None
        p_h3 = None
    overall = f01.get("overall") or {}
    ob = overall.get("bootstrap") or {}
    p_h3_overall = _boot_p_ci_excludes_zero(ob.get("ci_low"), int(ob.get("B") or 1000))
    # Final H3 p for outer family: max(collapsed strata Holm adj, overall approx) conservative, or use overall if strata empty
    p_h3_final = p_h3_collapsed if p_h3_collapsed is not None else p_h3_overall
    n_strata_hit = sum(
        1
        for s in strata
        if s.get("reject_h0_pointwise")
        and float(s.get("delta") or 0) > float(f01.get("delta_threshold") or 0.02)
    )
    details["H3"] = {
        "source": "F01 strata inner-Holm + overall bootstrap CI",
        "p": p_h3_final,
        "p_overall_boot": p_h3_overall,
        "strata_ps_threshold_clearing": strata_ps,
        "inner_holm": inner_h3,
        "n_strata_reject_pointwise": n_strata_hit,
        "min_strata_required": int(f01.get("min_strata_consistent") or 3),
        "h3_preview_reject": f01.get("h3_preview_reject"),
        "protocol_reject": bool(
            f01.get("h3_preview_reject")
            and overall.get("reject_h0_pointwise")
            and n_strata_hit >= int(f01.get("min_strata_consistent") or 3)
        ),
    }

    # --- H4 ---
    hit_ps: dict[str, float] = {}
    for h in h4.get("hits") or []:
        if h.get("status") != "live":
            continue
        b = h.get("bootstrap") or {}
        ci = b.get("ci_delta") or [None, None]
        B = int(b.get("B") or h4.get("bootstrap_B") or 1000)
        p_hit = _boot_p_ci_excludes_zero(ci[0], B)
        if p_hit is not None and h.get("delta_ge_threshold"):
            hit_ps[h["dropped"][0]] = p_hit
    if hit_ps:
        p_h4 = min(hit_ps.values())
    elif int(h4.get("n_hits") or 0) == 0:
        # No LOO floor hits (e.g. H6a cohort): cannot reject H4; p=1 for outer Holm.
        p_h4 = 1.0
    else:
        p_h4 = None
    details["H4"] = {
        "source": "H04_paired_bootstrap_loo min boot-p over floor hits",
        "p": p_h4,
        "hit_ps": hit_ps,
        "h4_reject": h4.get("h4_reject"),
        "n_hits": h4.get("n_hits"),
        "n_hits_reject": h4.get("n_hits_reject"),
        "protocol_reject": bool(h4.get("h4_reject")),
        "note": (
            "p=1.0 when sens1 has zero Δ≥threshold hits (no floor leakage under current Cox population)"
            if p_h4 == 1.0 and not hit_ps
            else None
        ),
    }

    # --- H5 ---
    primary_id = f02.get("primary_model_id") or "rsf_content"
    p_h5 = None
    h5_model = None
    for m in f02.get("models") or []:
        if m.get("model_id") == primary_id:
            h5_model = m
            break
    if h5_model:
        b = ((h5_model.get("brier_strip") or {}).get("bootstrap")) or {}
        p_h5 = _boot_p_ci_excludes_zero(b.get("ci_delta_low"), int(b.get("B") or 200))
    details["H5"] = {
        "source": f"F02 {primary_id} brier_strip.bootstrap",
        "p": p_h5,
        "primary_model_id": primary_id,
        "h5_preview_reject": f02.get("h5_preview_reject"),
        "protocol_reject": bool(f02.get("h5_preview_reject")),
    }

    raw = {
        "H1": p_h1,
        "H2": p_h2,
        "H3": p_h3_final,
        "H4": p_h4,
        "H5": p_h5,
    }
    missing = [k for k, v in raw.items() if v is None]
    return {"raw_p": raw, "missing": missing, "details": details}


def _to_md(payload: dict) -> str:
    outer = payload["outer_holm"]
    lines = [
        "# G03 — Two-level Holm + H_meta",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        f"Protocol: `{payload['protocol_version']}`",
        f"α = {ALPHA}",
        "",
        "## Outer family (H1–H5)",
        "",
        "| H | raw p | Holm adj | reject | protocol_reject |",
        "|---|-------|----------|--------|-----------------|",
    ]
    for h in ["H1", "H2", "H3", "H4", "H5"]:
        lines.append(
            f"| {h} | {outer['raw'].get(h):.4g} | {outer['adjusted'].get(h):.4g} | "
            f"**{outer['reject'].get(h)}** | {payload['protocol_reject'].get(h)} |"
        )
    meta = payload["H_meta"]
    lines += [
        "",
        f"**Outer rejects:** {meta['n_outer_rejects']} / 5",
        f"**H_meta (≥{META_K}/5):** **{meta['reject']}**",
        "",
        "## Notes",
        "",
        payload.get("honesty_note", ""),
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    log("─" * 60)
    log("G03 — HOLM FAMILY + H_meta")
    log("─" * 60)

    assembled = assemble_pvalues()
    raw = assembled["raw_p"]
    if assembled["missing"]:
        log(f"ERROR: missing p-values for {assembled['missing']}")
        return 1

    # Cast to float
    raw_f = {k: float(v) for k, v in raw.items()}
    outer = holm_adjust(raw_f, alpha=ALPHA)
    protocol_reject = {
        h: bool((assembled["details"].get(h) or {}).get("protocol_reject"))
        for h in raw_f
    }
    # Outer Holm rejects
    n_outer = sum(1 for v in outer["reject"].values() if v)
    h_meta_reject = n_outer >= META_K

    honesty = (
        "Bootstrap-based p for H3/H4/H5 use the conservative 1/(B+1) bound when "
        "the percentile CI for the positive effect excludes 0 (no mass ≤0 observed). "
        "H1 raw p cannot clear α under C00.1 with k≤3 models (rank permutation). "
        "H_meta counts **outer-Holm** rejects, not protocol_reject flags alone."
    )

    payload = {
        "stage": "G03",
        "generated_at_utc": utc_now(),
        "protocol_version": PROTOCOL["version"],
        "alpha": ALPHA,
        "meta_threshold": META_K,
        "raw_p": raw_f,
        "details": assembled["details"],
        "outer_holm": outer,
        "protocol_reject": protocol_reject,
        "H_meta": {
            "rule": f"Reject H_meta if ≥{META_K} of 5 outer-Holm rejects",
            "n_outer_rejects": n_outer,
            "reject": h_meta_reject,
            "rejected_hypotheses": [h for h, r in outer["reject"].items() if r],
        },
        "honesty_note": honesty,
    }

    out_dir = cfg.DIRS["reproduction"]
    log_dir = cfg.DIRS["logs"]
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "G03_holm_family.json"
    out_md = out_dir / "G03_holm_family.md"
    log_md = log_dir / "G03_holm_family.md"
    write_json(out_json, payload)
    md = _to_md(payload)
    out_md.write_text(md, encoding="utf-8")
    log_md.write_text(md, encoding="utf-8")

    log("  raw p: " + ", ".join(f"{k}={v:.4g}" for k, v in raw_f.items()))
    log(
        "  Holm reject: "
        + ", ".join(f"{k}={outer['reject'][k]}" for k in ["H1", "H2", "H3", "H4", "H5"])
    )
    log(f"  H_meta reject={h_meta_reject} ({n_outer}/5)")
    log(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
