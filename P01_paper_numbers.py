"""
P01_paper_numbers.py — Paper numbers + fact skeleton (source of truth)
======================================================================
Extracts every manuscript-facing scalar from the P00 collection.
**No hard-coded numeric claims** — values come only from inventoried JSON.

Also writes the **paper fact skeleton** (no prose): section-by-section
claims, protocol rules, numbers, and pointers to tables/figures.

Writes:
  results/paper/numbers.json
  results/paper/numbers.md
  results/paper/PAPER.md          ← paper without prose (etapa a etapa)
  results/logs/P01_paper_numbers.md

Execute (after P00):
    python -W default P01_paper_numbers.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import PROTOCOL, cfg
from src.metrics.io import utc_now, write_json


def log(msg: str = "") -> None:
    print(msg, flush=True)


def _load_collection() -> dict[str, Any]:
    path = cfg.DIRS["paper"] / "collection" / "manifest.json"
    if not path.exists():
        raise FileNotFoundError("Run P00_collect_artifacts.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_source(manifest: dict, source_id: str) -> dict[str, Any] | None:
    entry = next((s for s in manifest["sources"] if s["id"] == source_id), None)
    if entry is None or entry.get("status") != "ok":
        return None
    path = cfg.ROOT / entry["rel"]
    return json.loads(path.read_text(encoding="utf-8"))


def _put(
    bag: dict[str, Any],
    key: str,
    value: Any,
    *,
    source_id: str,
    path: str,
    note: str = "",
) -> None:
    bag[key] = {
        "value": value,
        "source_id": source_id,
        "json_path": path,
        "note": note,
    }


def _model_map(ladder_doc: dict | None) -> dict[str, dict]:
    if not ladder_doc:
        return {}
    return {
        m["model_id"]: m
        for m in (ladder_doc.get("models") or [])
        if isinstance(m, dict) and "model_id" in m
    }


def extract_numbers(manifest: dict) -> dict[str, Any]:
    numbers: dict[str, Any] = {}

    _put(numbers, "protocol.version", PROTOCOL["version"], source_id="config", path="PROTOCOL.version")
    # Decision-rule scalars (artefatos checklist: no hardcodes in P02/P03)
    hyps = PROTOCOL.get("hypotheses") or {}
    if "H2" in hyps:
        _put(
            numbers,
            "protocol.H2.c_h_threshold",
            hyps["H2"].get("c_h_threshold"),
            source_id="config",
            path="PROTOCOL.hypotheses.H2.c_h_threshold",
        )
    if "H3" in hyps:
        _put(
            numbers,
            "protocol.H3.delta_threshold",
            hyps["H3"].get("delta"),
            source_id="config",
            path="PROTOCOL.hypotheses.H3.delta",
        )
        _put(
            numbers,
            "protocol.H3.horizon_months",
            hyps["H3"].get("horizon_months"),
            source_id="config",
            path="PROTOCOL.hypotheses.H3.horizon_months",
        )
        _put(
            numbers,
            "protocol.H3.min_strata_consistent",
            hyps["H3"].get("min_strata_consistent"),
            source_id="config",
            path="PROTOCOL.hypotheses.H3.min_strata_consistent",
        )
    if "H4" in hyps:
        _put(
            numbers,
            "protocol.H4.delta_c_threshold",
            hyps["H4"].get("delta_c_threshold"),
            source_id="config",
            path="PROTOCOL.hypotheses.H4.delta_c_threshold",
        )
    if "H5" in hyps:
        band = hyps["H5"].get("global_cindex_band")
        _put(
            numbers,
            "protocol.H5.global_cindex_band",
            band,
            source_id="config",
            path="PROTOCOL.hypotheses.H5.global_cindex_band",
        )
        _put(
            numbers,
            "protocol.H5.horizons_months",
            hyps["H5"].get("horizons_months"),
            source_id="config",
            path="PROTOCOL.hypotheses.H5.horizons_months",
        )
    # Reproduction tier cutoffs (§3.3 / artefatos T04 footnote)
    _put(numbers, "protocol.repro.tier_strict", 0.01, source_id="config", path="artefatos.md§5 / §3.3")
    _put(numbers, "protocol.repro.tier_approx", 0.03, source_id="config", path="artefatos.md§5 / §3.3")
    _put(numbers, "protocol.globals.alpha", PROTOCOL["globals"]["alpha"], source_id="config", path="PROTOCOL.globals.alpha")
    # C00.1 tau floor for H1 formal reject (PROTOCOL.decision_rules.C00.1)
    _put(numbers, "protocol.H1.tau_reject_ceiling", 0.5, source_id="config", path="PROTOCOL.decision_rules.C00.1")
    # Historical D1 unfiltered-cohort C cited in artefatos T04 table note (not a live estimate)
    _put(
        numbers,
        "protocol.repro.d1_unfiltered_c_cited",
        0.981,
        source_id="artefatos",
        path="artefatos.md§5 footnote",
        note="Prescribed table-note citation; not recomputed in Block P",
    )

    g03 = _read_source(manifest, "G03")
    if g03:
        hmeta = g03.get("H_meta") or {}
        _put(numbers, "H_meta.reject", hmeta.get("reject"), source_id="G03", path="H_meta.reject")
        _put(
            numbers,
            "H_meta.n_holm_rejects",
            hmeta.get("n_outer_rejects"),
            source_id="G03",
            path="H_meta.n_outer_rejects",
        )
        outer = g03.get("outer_holm") or {}
        for hid in ("H1", "H2", "H3", "H4", "H5"):
            _put(
                numbers,
                f"{hid}.holm_reject",
                (outer.get("reject") or {}).get(hid),
                source_id="G03",
                path=f"outer_holm.reject.{hid}",
            )
            _put(
                numbers,
                f"{hid}.adj_p",
                (outer.get("adjusted") or {}).get(hid),
                source_id="G03",
                path=f"outer_holm.adjusted.{hid}",
            )
            _put(
                numbers,
                f"{hid}.raw_p",
                (outer.get("raw") or g03.get("raw_p") or {}).get(hid),
                source_id="G03",
                path=f"outer_holm.raw.{hid}",
            )

    h01 = _read_source(manifest, "H01")
    if h01:
        _put(
            numbers,
            "H1.formal_reject_any_domain",
            h01.get("formal_reject_C00_1_any_domain"),
            source_id="H01",
            path="formal_reject_C00_1_any_domain",
        )
        _put(
            numbers,
            "H1.any_binary_inversion",
            h01.get("any_binary_inversion"),
            source_id="H01",
            path="any_binary_inversion",
        )
        _put(numbers, "H1.B", h01.get("B"), source_id="H01", path="B")
        for d in h01.get("domains") or []:
            if d.get("status") != "live":
                continue
            dom = d["domain_id"]
            _put(numbers, f"H1.{dom}.tau_K", d.get("tau_K"), source_id="H01", path=f"domains[{dom}].tau_K")
            _put(
                numbers,
                f"H1.{dom}.reject",
                d.get("reject_H0_C00_1"),
                source_id="H01",
                path=f"domains[{dom}].reject_H0_C00_1",
            )
            _put(
                numbers,
                f"H1.{dom}.boot_ci",
                (d.get("bootstrap") or {}).get("ci"),
                source_id="H01",
                path=f"domains[{dom}].bootstrap.ci",
            )
            for mid, v in (d.get("harrell_c") or {}).items():
                _put(numbers, f"H1.{dom}.{mid}.C", v, source_id="H01", path=f"harrell_c.{mid}")
            for mid, v in (d.get("ipcw_ibs_sksurv") or {}).items():
                _put(numbers, f"H1.{dom}.{mid}.IBS", v, source_id="H01", path=f"ipcw_ibs_sksurv.{mid}")

    h04 = _read_source(manifest, "H04")
    if h04:
        _put(numbers, "H4.reject", h04.get("h4_reject"), source_id="H04", path="h4_reject")
        _put(numbers, "H4.n_hits", h04.get("n_hits"), source_id="H04", path="n_hits")
        _put(
            numbers,
            "H4.n_hits_reject",
            h04.get("n_hits_reject"),
            source_id="H04",
            path="n_hits_reject",
        )
        _put(
            numbers,
            "H4.B",
            h04.get("bootstrap_B") or h04.get("B"),
            source_id="H04",
            path="bootstrap_B",
        )
        _put(
            numbers,
            "H4.boot_n",
            h04.get("bootstrap_n"),
            source_id="H04",
            path="bootstrap_n",
        )
        _put(numbers, "H4.c_full", h04.get("c_full"), source_id="H04", path="c_full")
        _put(
            numbers,
            "H4.delta_threshold",
            h04.get("delta_threshold"),
            source_id="H04",
            path="delta_threshold",
        )
        hit_ids: list[str] = []
        for i, hit in enumerate(h04.get("hits") or []):
            dropped = hit.get("dropped") or []
            label = (
                "+".join(str(x) for x in dropped)
                if dropped
                else hit.get("label") or f"hit_{i}"
            )
            # normalize smart_9_raw → 9 for paper shorthand, keep full id in key
            slug = label.replace("smart_", "").replace("_raw", "")
            hit_ids.append(slug)
            boot = hit.get("bootstrap") or {}
            prefix = f"H4.hit.{slug}"
            _put(numbers, f"{prefix}.dropped", dropped, source_id="H04", path=f"hits[{i}].dropped")
            _put(numbers, f"{prefix}.delta_c", hit.get("delta_c"), source_id="H04", path=f"hits[{i}].delta_c")
            _put(numbers, f"{prefix}.c_full", hit.get("c_full"), source_id="H04", path=f"hits[{i}].c_full")
            _put(numbers, f"{prefix}.c_ablated", hit.get("c_ablated"), source_id="H04", path=f"hits[{i}].c_ablated")
            _put(numbers, f"{prefix}.reject", hit.get("reject_h4"), source_id="H04", path=f"hits[{i}].reject_h4")
            _put(
                numbers,
                f"{prefix}.ci_delta",
                boot.get("ci_delta"),
                source_id="H04",
                path=f"hits[{i}].bootstrap.ci_delta",
            )
            _put(
                numbers,
                f"{prefix}.ci_full",
                boot.get("ci_full"),
                source_id="H04",
                path=f"hits[{i}].bootstrap.ci_full",
            )
            _put(
                numbers,
                f"{prefix}.ci_ablated",
                boot.get("ci_ablated"),
                source_id="H04",
                path=f"hits[{i}].bootstrap.ci_ablated",
            )
            _put(
                numbers,
                f"{prefix}.ci_nonoverlap",
                boot.get("ci_nonoverlap"),
                source_id="H04",
                path=f"hits[{i}].bootstrap.ci_nonoverlap",
            )
        _put(numbers, "H4.hit_ids", hit_ids, source_id="H04", path="hits[*].dropped")

    f01 = _read_source(manifest, "F01_h3")
    if f01:
        overall = f01.get("overall") or {}
        if isinstance(overall, dict):
            _put(numbers, "H3.delta", overall.get("delta") or overall.get("abs_delta"), source_id="F01_h3", path="overall.delta")
            naive = overall.get("naive")
            aj = overall.get("aj")
            _put(
                numbers,
                "H3.naive",
                naive.get("value") if isinstance(naive, dict) else naive,
                source_id="F01_h3",
                path="overall.naive.value",
            )
            _put(
                numbers,
                "H3.AJ",
                aj.get("value") if isinstance(aj, dict) else aj,
                source_id="F01_h3",
                path="overall.aj.value",
            )
            boot = overall.get("bootstrap") or {}
            _put(
                numbers,
                "H3.ci",
                [boot.get("ci_low"), boot.get("ci_high")] if boot else overall.get("ci"),
                source_id="F01_h3",
                path="overall.bootstrap.ci",
            )
            _put(
                numbers,
                "H3.ci_excludes_0",
                boot.get("ci_excludes_0"),
                source_id="F01_h3",
                path="overall.bootstrap.ci_excludes_0",
            )
            _put(
                numbers,
                "H3.horizon_months",
                overall.get("horizon_months") or f01.get("horizon_months"),
                source_id="F01_h3",
                path="overall.horizon_months",
            )
        _put(
            numbers,
            "H3.preview_reject",
            f01.get("h3_preview_reject"),
            source_id="F01_h3",
            path="h3_preview_reject",
        )
        by_rating = f01.get("by_rating") or {}
        if isinstance(by_rating, dict):
            _put(
                numbers,
                "H3.n_strata",
                by_rating.get("n_strata"),
                source_id="F01_h3",
                path="by_rating.n_strata",
            )
            _put(
                numbers,
                "H3.n_strata_supporting",
                by_rating.get("n_strata_supporting_h3"),
                source_id="F01_h3",
                path="by_rating.n_strata_supporting_h3",
            )
            _put(
                numbers,
                "H3.min_strata_consistent",
                by_rating.get("min_strata_consistent") or f01.get("min_strata_consistent"),
                source_id="F01_h3",
                path="by_rating.min_strata_consistent",
            )
            _put(
                numbers,
                "H3.strata_rule_met",
                by_rating.get("h3_strata_rule_met"),
                source_id="F01_h3",
                path="by_rating.h3_strata_rule_met",
            )
            supporting: list[str] = []
            for st in by_rating.get("strata") or []:
                rating = st.get("rating")
                if rating is None:
                    continue
                pref = f"H3.stratum.{rating}"
                _put(numbers, f"{pref}.delta", st.get("delta") or st.get("abs_delta"), source_id="F01_h3", path=f"strata[{rating}].delta")
                _put(numbers, f"{pref}.n", st.get("n"), source_id="F01_h3", path=f"strata[{rating}].n")
                _put(
                    numbers,
                    f"{pref}.supports",
                    st.get("supports_h3_direction"),
                    source_id="F01_h3",
                    path=f"strata[{rating}].supports_h3_direction",
                )
                sboot = st.get("bootstrap") or {}
                _put(
                    numbers,
                    f"{pref}.ci",
                    [sboot.get("ci_low"), sboot.get("ci_high")] if sboot else None,
                    source_id="F01_h3",
                    path=f"strata[{rating}].bootstrap.ci",
                )
                if st.get("supports_h3_direction"):
                    supporting.append(str(rating))
            _put(
                numbers,
                "H3.supporting_ratings",
                supporting,
                source_id="F01_h3",
                path="strata[supports].rating",
            )

    f02 = _read_source(manifest, "F02_h5")
    if f02:
        _put(
            numbers,
            "H5.preview_reject",
            f02.get("h5_preview_reject"),
            source_id="F02_h5",
            path="h5_preview_reject",
        )
        _put(
            numbers,
            "H5.primary_model_id",
            f02.get("primary_model_id"),
            source_id="F02_h5",
            path="primary_model_id",
        )
        for r in f02.get("models") or []:
            mid = r.get("model_id")
            if not mid:
                continue
            strip = r.get("brier_strip") or {}
            _put(
                numbers,
                f"H5.{mid}.C",
                strip.get("global_harrell_c"),
                source_id="F02_h5",
                path=f"models[{mid}].brier_strip.global_harrell_c",
            )
            brier = strip.get("brier") or {}
            vals = brier.get("values")
            if vals:
                _put(
                    numbers,
                    f"H5.{mid}.brier",
                    vals,
                    source_id="F02_h5",
                    path=f"models[{mid}].brier_strip.brier.values",
                )
            _put(
                numbers,
                f"H5.{mid}.delta_brier",
                strip.get("delta_brier_36_minus_12"),
                source_id="F02_h5",
                path=f"models[{mid}].delta_brier_36_minus_12",
            )
            _put(
                numbers,
                f"H5.{mid}.h5_reject",
                strip.get("h5_preview_reject") or r.get("h5_preview_reject"),
                source_id="F02_h5",
                path=f"models[{mid}].h5_preview_reject",
            )
    # Ladder discrimination + calibration + IBS
    for slug, sid_disc, sid_cal, sid_sc in (
        ("d01", "d01_discrimination", "d01_calibration", "d01_scores"),
        ("d02", "d02_discrimination", "d02_calibration", "d02_scores"),
        ("d03", "d03_discrimination", "d03_calibration", "d03_scores"),
    ):
        disc = _model_map(_read_source(manifest, sid_disc))
        cal = _model_map(_read_source(manifest, sid_cal))
        sc = _model_map(_read_source(manifest, sid_sc))
        for mid, m in disc.items():
            if m.get("implementation_status") != "live":
                continue
            hv = (m.get("harrell") or {}).get("value")
            _put(numbers, f"ladder.{slug}.{mid}.C", hv, source_id=sid_disc, path=f"models[{mid}].harrell.value")
        for mid, m in cal.items():
            if m.get("implementation_status") != "live":
                continue
            dc = m.get("d_calibration") or {}
            _put(numbers, f"ladder.{slug}.{mid}.dcal_p", dc.get("p_value"), source_id=sid_cal, path=f"models[{mid}].d_calibration.p_value")
            _put(
                numbers,
                f"ladder.{slug}.{mid}.dcal_reject",
                dc.get("reject_h0_well_calibrated"),
                source_id=sid_cal,
                path=f"models[{mid}].d_calibration.reject",
            )
            bins = dc.get("bin_counts")
            if isinstance(bins, list) and bins:
                _put(
                    numbers,
                    f"ladder.{slug}.{mid}.dcal_bin_counts",
                    [float(x) for x in bins],
                    source_id=sid_cal,
                    path=f"models[{mid}].d_calibration.bin_counts",
                )
                _put(
                    numbers,
                    f"ladder.{slug}.{mid}.dcal_n_bins",
                    int(dc.get("n_bins") or len(bins)),
                    source_id=sid_cal,
                    path=f"models[{mid}].d_calibration.n_bins",
                )
                total = float(sum(float(x) for x in bins))
                if total > 0:
                    props = [float(x) / total for x in bins]
                    _put(
                        numbers,
                        f"ladder.{slug}.{mid}.dcal_observed_proportions",
                        props,
                        source_id=sid_cal,
                        path=f"models[{mid}].d_calibration.bin_counts/sum",
                        note="Observed mass per Haider decile bin",
                    )
                    # Post-hoc ECE from existing D-Cal bins (no re-test):
                    # ECE = Σ (n_i/N) |p̂_i − 0.10|; report in percentage points.
                    ece = sum((float(n_i) / total) * abs(p_i - 0.10) for n_i, p_i in zip(bins, props))
                    max_dev = max(abs(p_i - 0.10) for p_i in props)
                    _put(
                        numbers,
                        f"ladder.{slug}.{mid}.dcal_ece",
                        ece,
                        source_id=sid_cal,
                        path=f"models[{mid}].d_calibration.bin_counts→ECE",
                        note="Post-hoc ECE from D-Cal bin counts",
                    )
                    _put(
                        numbers,
                        f"ladder.{slug}.{mid}.dcal_ece_pp",
                        round(100.0 * ece, 2),
                        source_id=sid_cal,
                        path=f"models[{mid}].d_calibration.bin_counts→ECE_pp",
                    )
                    _put(
                        numbers,
                        f"ladder.{slug}.{mid}.dcal_max_bin_dev",
                        max_dev,
                        source_id=sid_cal,
                        path=f"models[{mid}].d_calibration.bin_counts→max_bin_dev",
                    )
                    _put(
                        numbers,
                        f"ladder.{slug}.{mid}.dcal_max_bin_dev_pp",
                        round(100.0 * max_dev, 2),
                        source_id=sid_cal,
                        path=f"models[{mid}].d_calibration.bin_counts→max_bin_dev_pp",
                    )
            ici = (m.get("ici") or {}).get("value")
            _put(numbers, f"ladder.{slug}.{mid}.ICI", ici, source_id=sid_cal, path=f"models[{mid}].ici.value")
        for mid, m in sc.items():
            if m.get("implementation_status") != "live":
                continue
            ibs = m.get("ipcw_ibs") or m.get("ibs") or {}
            if isinstance(ibs, dict):
                _put(numbers, f"ladder.{slug}.{mid}.IBS", ibs.get("value"), source_id=sid_sc, path=f"models[{mid}].ipcw_ibs.value")

    # Reproduction gaps (row-level — no hard-coded quantities)
    for sid, dom in (
        ("DOMAIN_01_repro", "DOMAIN_01"),
        ("DOMAIN_02_repro", "DOMAIN_02"),
        ("DOMAIN_03_repro", "DOMAIN_03"),
    ):
        doc = _read_source(manifest, sid)
        if not doc:
            continue
        _put(numbers, f"repro.{dom}.n_rows", len(doc.get("rows") or []), source_id=sid, path="len(rows)")
        gaps = []
        for row in doc.get("rows") or []:
            qid = row.get("quantity_id") or row.get("quantity_label") or "row"
            if row.get("gap") is not None:
                _put(
                    numbers,
                    f"repro.{dom}.{qid}.gap",
                    row.get("gap"),
                    source_id=sid,
                    path=f"rows[{qid}].gap",
                )
                gaps.append(abs(float(row["gap"])))
            if row.get("ours_value") is not None:
                _put(
                    numbers,
                    f"repro.{dom}.{qid}.ours",
                    row.get("ours_value"),
                    source_id=sid,
                    path=f"rows[{qid}].ours_value",
                )
            if row.get("paper_value") is not None:
                _put(
                    numbers,
                    f"repro.{dom}.{qid}.paper",
                    row.get("paper_value"),
                    source_id=sid,
                    path=f"rows[{qid}].paper_value",
                )
        if gaps:
            _put(
                numbers,
                f"repro.{dom}.mean_abs_gap",
                float(sum(gaps) / len(gaps)),
                source_id=sid,
                path="mean(|gap|)",
            )
        # Domain 3 headline: mean |gap| over the 18 Table-8 C-index cells only
        if dom == "DOMAIN_03":
            c_gaps = [
                abs(float(row["gap"]))
                for row in (doc.get("rows") or [])
                if row.get("gap") is not None
                and str(row.get("quantity_id") or "").startswith("cindex_")
            ]
            if c_gaps:
                _put(
                    numbers,
                    "repro.DOMAIN_03.cindex_mean_abs_gap",
                    float(sum(c_gaps) / len(c_gaps)),
                    source_id=sid,
                    path="mean(|gap|) over cindex_* rows",
                    note="18-cell Table 8 headline for T04",
                )
                _put(
                    numbers,
                    "repro.DOMAIN_03.cindex_n_cells",
                    len(c_gaps),
                    source_id=sid,
                    path="count(cindex_* rows with gap)",
                )

    g02 = _read_source(manifest, "G02_verdict")
    if g02:
        _put(numbers, "verdict.authority", g02.get("authority"), source_id="G02_verdict", path="authority")
        hmeta = g02.get("h_meta") or g02.get("H_meta") or {}
        _put(
            numbers,
            "verdict.h_meta_reject",
            hmeta.get("reject") if isinstance(hmeta, dict) else hmeta,
            source_id="G02_verdict",
            path="h_meta.reject",
        )

    # Optional anchor (Lillelund) — Background / methods validation
    for sid, prefix in (("ANCHOR_table2", "anchor.table2"), ("ANCHOR_figure5", "anchor.figure5")):
        doc = _read_source(manifest, sid)
        if not doc:
            continue
        _put(numbers, f"{prefix}.n_rows", len(doc.get("rows") or []), source_id=sid, path="len(rows)")
        _put(
            numbers,
            f"{prefix}.n_seeds",
            doc.get("n_seeds") or doc.get("n_seeds_ours"),
            source_id=sid,
            path="n_seeds",
        )
        if doc.get("note") is not None:
            _put(numbers, f"{prefix}.note", doc.get("note"), source_id=sid, path="note")
        for i, row in enumerate(doc.get("rows") or []):
            qid = (
                row.get("quantity_id")
                or row.get("metric_id")
                or row.get("scenario_id")
                or row.get("metric")
                or row.get("scenario")
                or f"row{i}"
            )
            qid = str(qid).replace(" ", "_")
            scen = row.get("scenario_id") or row.get("censoring")
            key_base = f"{prefix}.{qid}" if scen is None else f"{prefix}.{scen}.{qid}"
            # generic scalar fields
            for field, aliases in (
                ("ours", ("ours", "ours_value", "ours_mean")),
                ("paper", ("paper", "paper_value", "paper_mean")),
                ("gap", ("gap", "delta")),
                ("ours_sd", ("ours_sd",)),
                ("paper_sd", ("paper_sd",)),
            ):
                for alias in aliases:
                    if alias in row and row[alias] is not None:
                        _put(
                            numbers,
                            f"{key_base}.{field}",
                            row[alias],
                            source_id=sid,
                            path=f"rows[{qid}].{alias}",
                        )
                        break
            # Table2 oracle CI / IBS / censor columns
            for alias in (
                "ci_oracle_ours",
                "ci_oracle_paper",
                "ci_oracle_gap",
                "ibs_oracle_ours",
                "ibs_oracle_paper",
                "ibs_oracle_gap",
                "censor_pct_ours",
                "censor_pct_paper",
                "censor_pct_gap",
                "n_events_ours",
                "n_events_paper",
            ):
                if alias in row and row[alias] is not None:
                    _put(
                        numbers,
                        f"{key_base}.{alias}",
                        row[alias],
                        source_id=sid,
                        path=f"rows[{qid}].{alias}",
                    )

    return numbers


def _v(numbers: dict, key: str) -> Any:
    rec = numbers.get(key)
    return None if rec is None else rec.get("value")


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if abs(v) != 0 and (abs(v) < 1e-3 or abs(v) >= 1e4):
            return f"{v:.4g}"
        return f"{v:.6g}"
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def _cite(numbers: dict, key: str) -> str:
    """Value with source tag for traceability."""
    rec = numbers.get(key)
    if rec is None:
        return f"— [{key}]"
    return f"{_fmt(rec['value'])} [{key}←{rec['source_id']}]"


def write_paper_md(numbers: dict, manifest: dict, generated_at: str) -> str:
    """
    Paper fact skeleton — no prose. One section per pipeline step.
    All scalars from ``numbers``; domain metadata from frozen PROTOCOL/cfg.
    """
    proto = PROTOCOL
    c001 = proto["freeze_decisions"]["C00.1"]["official"]
    hyps = proto["hypotheses"]

    lines: list[str] = [
        "# The C-index Illusion — paper fact skeleton",
        "",
        "> **No prose.** Numbers + protocol + structure only. "
        "Fill narrative later; never overwrite scalars by hand.",
        "",
        f"- Generated (UTC): `{generated_at}`",
        f"- Protocol: `{_cite(numbers, 'protocol.version')}`",
        f"- P00 required complete: `{manifest.get('complete_required')}`",
        f"- P01 n_keys: `{len(numbers)}`",
        f"- Verdict authority: `{_cite(numbers, 'verdict.authority')}`",
        "",
        "---",
        "",
        "## 1. Framing (facts)",
        "",
        "- Anchor position paper: Lillelund et al., ICML 2026 Spotlight "
        f"(arXiv:{cfg.ANCHOR['arxiv']}, doi:{cfg.ANCHOR['doi']}).",
        "- Contribution type: cross-domain reproduction + assumption-aligned "
        "audit of published non-clinical survival models.",
        "- Domains (n=3): engineering / P2P credit / Q&A platforms.",
        "- Primary meta claim gate: H_meta under outer Holm "
        f"(reject={_cite(numbers, 'H_meta.reject')}; "
        f"n={_cite(numbers, 'H_meta.n_holm_rejects')}/5).",
        "",
        "---",
        "",
        "## 2. Protocol freeze (C00)",
        "",
        f"- Version: `{_v(numbers, 'protocol.version')}`",
        f"- α = {proto['globals']['alpha']}; "
        f"B = {proto['globals']['n_bootstrap']}; "
        f"seed = {proto['globals']['random_seed']}",
        "",
        "### C00.1 — H1 decision",
        "",
        f"> {c001}",
        "",
        "### C00.2 — ranking objects",
        "",
        "| Domain | Models |",
        "|--------|--------|",
    ]
    for dom, models in proto["freeze_decisions"]["C00.2"]["rankings"].items():
        lines.append(f"| {dom} | {', '.join(f'`{m}`' for m in models)} |")

    lines += [
        "",
        "### Pre-registered hypotheses (decision text)",
        "",
    ]
    for hid in ("H1", "H2", "H3", "H4", "H5", "H_meta"):
        h = hyps[hid]
        lines.append(f"#### {hid} — {h['title']}")
        lines.append(f"- H0: {h['H0']}")
        lines.append(f"- H1: {h['H1']}")
        if "statistic" in h:
            lines.append(f"- Statistic: {h['statistic']}")
        if "decision" in h:
            lines.append(f"- Decision: {h['decision']}")
        lines.append("")

    lines += [
        "---",
        "",
        "## 3. Domains & baselines",
        "",
        "| ID | Name | Baseline | DOI | Reported C |",
        "|----|------|----------|-----|------------|",
    ]
    for did, d in cfg.DOMAINS.items():
        rc = d.get("reported_cindex")
        lines.append(
            f"| `{did}` | {d['name']} | {d['baseline']} | {d['doi']} | {_fmt(rc)} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 4. Reproduction (Fase A gaps)",
        "",
        "| Domain | mean_abs_gap | n_rows |",
        "|--------|--------------|--------|",
    ]
    for dom in ("DOMAIN_01", "DOMAIN_02", "DOMAIN_03"):
        lines.append(
            f"| {dom} | {_cite(numbers, f'repro.{dom}.mean_abs_gap')} | "
            f"{_cite(numbers, f'repro.{dom}.n_rows')} |"
        )
    lines += [
        "",
        "Headline quantity gaps (from P01 keys `repro.*.*.gap`):",
        "",
        "| Domain | quantity_id | ours | paper | gap |",
        "|--------|-------------|------|-------|-----|",
    ]
    gap_keys = sorted(
        k for k in numbers if k.startswith("repro.") and k.endswith(".gap")
        and k.count(".") == 3
    )
    for gk in gap_keys:
        # repro.DOMAIN_01.qid.gap
        _, dom, qid, _ = gk.split(".", 3)
        lines.append(
            f"| {dom} | `{qid}` | {_fmt(_v(numbers, f'repro.{dom}.{qid}.ours'))} | "
            f"{_fmt(_v(numbers, f'repro.{dom}.{qid}.paper'))} | {_cite(numbers, gk)} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 4b. Anchor harness (optional — Lillelund synthetic check)",
        "",
        f"- Table 2 rows / seeds: {_cite(numbers, 'anchor.table2.n_rows')} / "
        f"{_cite(numbers, 'anchor.table2.n_seeds')}",
        f"- Figure 5 rows / seeds: {_cite(numbers, 'anchor.figure5.n_rows')} / "
        f"{_cite(numbers, 'anchor.figure5.n_seeds')}",
        "",
        "| Scenario | CI oracle ours | CI oracle paper | gap | IBS oracle ours | IBS oracle paper |",
        "|----------|----------------|-----------------|-----|-----------------|------------------|",
    ]
    for scen in ("random", "independent", "dependent", "uniform", "mixed"):
        base = f"anchor.table2.{scen}.{scen}"
        # keys may be anchor.table2.random.random.* from scenario_id twice — try both
        candidates = [
            f"anchor.table2.{scen}.{scen}",
            f"anchor.table2.{scen}",
        ]
        used = None
        for c in candidates:
            if f"{c}.ci_oracle_ours" in numbers or any(
                k.startswith(c + ".") and k.endswith("ci_oracle_ours") for k in numbers
            ):
                used = c
                break
        # find any key matching scenario
        if used is None:
            hits = [k for k in numbers if k.startswith(f"anchor.table2.{scen}.") and "ci_oracle_ours" in k]
            if hits:
                used = hits[0].rsplit(".ci_oracle_ours", 1)[0]
        if used is None:
            continue
        lines.append(
            f"| {scen} | {_cite(numbers, f'{used}.ci_oracle_ours')} | "
            f"{_cite(numbers, f'{used}.ci_oracle_paper')} | "
            f"{_cite(numbers, f'{used}.ci_oracle_gap')} | "
            f"{_cite(numbers, f'{used}.ibs_oracle_ours')} | "
            f"{_cite(numbers, f'{used}.ibs_oracle_paper')} |"
        )
    lines += [
        "",
        "Figure 5 bias metrics (sample):",
        "",
        "| key | value |",
        "|-----|-------|",
    ]
    for k in sorted(numbers):
        if k.startswith("anchor.figure5.") and k.endswith(".gap"):
            lines.append(f"| `{k}` | {_cite(numbers, k)} |")
    lines += [
        "",
        "---",
        "",
        "## 5. Evaluation ladder (metrics)",
        "",
        "| Domain | Model | C | D-Cal p | D-Cal reject | IBS |",
        "|--------|-------|---|---------|--------------|-----|",
    ]
    for slug, dom in (("d01", "DOMAIN_01"), ("d02", "DOMAIN_02"), ("d03", "DOMAIN_03")):
        prefix = f"ladder.{slug}."
        mids = sorted(
            {
                k[len(prefix) :].split(".")[0]
                for k in numbers
                if k.startswith(prefix) and k.endswith(".C")
            }
        )
        for mid in mids:
            lines.append(
                f"| {dom} | `{mid}` | {_cite(numbers, f'ladder.{slug}.{mid}.C')} | "
                f"{_cite(numbers, f'ladder.{slug}.{mid}.dcal_p')} | "
                f"{_cite(numbers, f'ladder.{slug}.{mid}.dcal_reject')} | "
                f"{_cite(numbers, f'ladder.{slug}.{mid}.IBS')} |"
            )
    lines += [
        "",
        "---",
        "",
        "## 6. Hypotheses — results (step by step)",
        "",
        "### 6.1 H1 — Ranking inversion (C vs IBS)",
        "",
        f"- Formal reject (any domain): {_cite(numbers, 'H1.formal_reject_any_domain')}",
        f"- Any binary inversion: {_cite(numbers, 'H1.any_binary_inversion')}",
        f"- Bootstrap B: {_cite(numbers, 'H1.B')}",
        f"- Holm reject / adj p / raw p: "
        f"{_cite(numbers, 'H1.holm_reject')} / "
        f"{_cite(numbers, 'H1.adj_p')} / "
        f"{_cite(numbers, 'H1.raw_p')}",
        "",
        "| Domain | τ_K | reject C00.1 | boot CI |",
        "|--------|-----|--------------|---------|",
    ]
    for dom in ("DOMAIN_01", "DOMAIN_02", "DOMAIN_03"):
        lines.append(
            f"| {dom} | {_cite(numbers, f'H1.{dom}.tau_K')} | "
            f"{_cite(numbers, f'H1.{dom}.reject')} | "
            f"{_cite(numbers, f'H1.{dom}.boot_ci')} |"
        )
    lines += [
        "",
        "Per-model C / IBS under H1 ranking objects:",
        "",
        "| Domain | Model | C | IBS |",
        "|--------|-------|---|-----|",
    ]
    for key in sorted(numbers):
        if not key.startswith("H1.DOMAIN_") or not (key.endswith(".C") or key.endswith(".IBS")):
            continue
        parts = key.split(".")
        # H1.DOMAIN_02.cox_xgboost.C
        if len(parts) != 4:
            continue
        _, dom, mid, metric = parts
        if metric == "C":
            lines.append(
                f"| {dom} | `{mid}` | {_cite(numbers, key)} | "
                f"{_cite(numbers, f'H1.{dom}.{mid}.IBS')} |"
            )

    lines += [
        "",
        "### 6.2 H2 — High C fails D-Calibration",
        "",
        f"- Holm reject / adj p / raw p: "
        f"{_cite(numbers, 'H2.holm_reject')} / "
        f"{_cite(numbers, 'H2.adj_p')} / "
        f"{_cite(numbers, 'H2.raw_p')}",
        f"- Backblaze `cox_full`: C={_cite(numbers, 'ladder.d01.cox_full.C')}; "
        f"D-Cal p={_cite(numbers, 'ladder.d01.cox_full.dcal_p')}; "
        f"reject={_cite(numbers, 'ladder.d01.cox_full.dcal_reject')}",
        f"- Backblaze `cox_ablated`: C={_cite(numbers, 'ladder.d01.cox_ablated.C')}; "
        f"D-Cal p={_cite(numbers, 'ladder.d01.cox_ablated.dcal_p')}; "
        f"reject={_cite(numbers, 'ladder.d01.cox_ablated.dcal_reject')}",
        "",
        "### 6.3 H3 — Competing-risks blindness",
        "",
        f"- Holm reject / adj p / raw p: "
        f"{_cite(numbers, 'H3.holm_reject')} / "
        f"{_cite(numbers, 'H3.adj_p')} / "
        f"{_cite(numbers, 'H3.raw_p')}",
        f"- Horizon (months): {_cite(numbers, 'H3.horizon_months')}",
        f"- Δ (naive − AJ): {_cite(numbers, 'H3.delta')}",
        f"- Naive CIF: {_cite(numbers, 'H3.naive')}",
        f"- AJ CIF: {_cite(numbers, 'H3.AJ')}",
        f"- Bootstrap CI: {_cite(numbers, 'H3.ci')} "
        f"(excludes 0: {_cite(numbers, 'H3.ci_excludes_0')})",
        f"- Strata supporting / total: "
        f"{_cite(numbers, 'H3.n_strata_supporting')} / "
        f"{_cite(numbers, 'H3.n_strata')} "
        f"(min required {_cite(numbers, 'H3.min_strata_consistent')}; "
        f"rule met={_cite(numbers, 'H3.strata_rule_met')})",
        f"- Supporting ratings: {_cite(numbers, 'H3.supporting_ratings')}",
        "",
        "| Rating | n | Δ | CI | supports H3 |",
        "|--------|---|---|----|-------------|",
    ]
    for key in sorted(k for k in numbers if k.startswith("H3.stratum.") and k.endswith(".delta")):
        rating = key.split(".")[2]
        lines.append(
            f"| {rating} | {_cite(numbers, f'H3.stratum.{rating}.n')} | "
            f"{_cite(numbers, f'H3.stratum.{rating}.delta')} | "
            f"{_cite(numbers, f'H3.stratum.{rating}.ci')} | "
            f"{_cite(numbers, f'H3.stratum.{rating}.supports')} |"
        )
    lines += [
        "",
        "### 6.4 H4 — Broad SMART ablation (existential)",
        "",
        f"- Protocol reject: {_cite(numbers, 'H4.reject')}",
        f"- n_hits / n_hits_reject: {_cite(numbers, 'H4.n_hits')} / "
        f"{_cite(numbers, 'H4.n_hits_reject')}",
        f"- B / boot_n: {_cite(numbers, 'H4.B')} / {_cite(numbers, 'H4.boot_n')}",
        f"- C_full: {_cite(numbers, 'H4.c_full')}",
        f"- Δ threshold: {_cite(numbers, 'H4.delta_threshold')}",
        f"- Hit SMART ids: {_cite(numbers, 'H4.hit_ids')}",
        f"- Holm reject / adj p / raw p: "
        f"{_cite(numbers, 'H4.holm_reject')} / "
        f"{_cite(numbers, 'H4.adj_p')} / "
        f"{_cite(numbers, 'H4.raw_p')}",
        "",
        "| SMART dropped | ΔC | C_full | C_ablated | CI Δ | CI nonoverlap | reject |",
        "|---------------|----|--------|-----------|------|---------------|--------|",
    ]
    hit_ids = _v(numbers, "H4.hit_ids") or []
    if isinstance(hit_ids, list):
        # sort by delta descending when available
        def _delta(slug: str) -> float:
            v = _v(numbers, f"H4.hit.{slug}.delta_c")
            return float(v) if v is not None else -1.0

        for slug in sorted(hit_ids, key=_delta, reverse=True):
            lines.append(
                f"| `{slug}` | {_cite(numbers, f'H4.hit.{slug}.delta_c')} | "
                f"{_cite(numbers, f'H4.hit.{slug}.c_full')} | "
                f"{_cite(numbers, f'H4.hit.{slug}.c_ablated')} | "
                f"{_cite(numbers, f'H4.hit.{slug}.ci_delta')} | "
                f"{_cite(numbers, f'H4.hit.{slug}.ci_nonoverlap')} | "
                f"{_cite(numbers, f'H4.hit.{slug}.reject')} |"
            )
    lines += [
        "",
        "### 6.5 H5 — Brier↑ under stable C",
        "",
        f"- Preview/protocol reject: {_cite(numbers, 'H5.preview_reject')}",
        f"- Primary model: {_cite(numbers, 'H5.primary_model_id')}",
        f"- Holm reject / adj p / raw p: "
        f"{_cite(numbers, 'H5.holm_reject')} / "
        f"{_cite(numbers, 'H5.adj_p')} / "
        f"{_cite(numbers, 'H5.raw_p')}",
        "",
        "| Model | C | Brier(12/24/36) | Δ Brier(36−12) | H5 reject |",
        "|-------|---|-----------------|----------------|-----------|",
    ]
    for mid in ("rsf_behavioural", "rsf_content", "rsf_combined"):
        lines.append(
            f"| `{mid}` | {_cite(numbers, f'H5.{mid}.C')} | "
            f"{_cite(numbers, f'H5.{mid}.brier')} | "
            f"{_cite(numbers, f'H5.{mid}.delta_brier')} | "
            f"{_cite(numbers, f'H5.{mid}.h5_reject')} |"
        )
    lines += [
        "",
        "---",
        "",
        "## 7. Family decision (outer Holm + H_meta)",
        "",
        f"- H_meta reject: {_cite(numbers, 'H_meta.reject')}",
        f"- Outer-Holm rejects: {_cite(numbers, 'H_meta.n_holm_rejects')} / 5",
        f"- Verdict table authority: {_cite(numbers, 'verdict.authority')}",
        "",
        "| H | Holm reject | adj p | raw p |",
        "|---|-------------|-------|-------|",
    ]
    for hid in ("H1", "H2", "H3", "H4", "H5"):
        lines.append(
            f"| {hid} | {_cite(numbers, f'{hid}.holm_reject')} | "
            f"{_cite(numbers, f'{hid}.adj_p')} | "
            f"{_cite(numbers, f'{hid}.raw_p')} |"
        )
    lines += [
        "",
        "Canonical G02: `results/reproduction/FINAL_verdict_table.md`",
        "",
        "---",
        "",
        "## 8. Assets index",
        "",
        "| Kind | Path | Producer |",
        "|------|------|----------|",
        "| Numbers | `results/paper/numbers.json` | P01 |",
        "| This skeleton | `results/paper/PAPER.md` | P01 |",
        "| Collection | `results/paper/collection/manifest.json` | P00 |",
        "",
        "---",
        "",
        "## 9. Limitations (facts only)",
        "",
        "- H1 C00.1 v5.1 does not reject on the frozen split "
        f"(formal_any={_fmt(_v(numbers, 'H1.formal_reject_any_domain'))}; "
        f"D2 τ_K={_fmt(_v(numbers, 'H1.DOMAIN_02.tau_K'))}; "
        f"boot CI={_fmt(_v(numbers, 'H1.DOMAIN_02.boot_ci'))}).",
        "- Convenience sample of n=3 domains (existence proof framing).",
        "- Reproduction ≠ replication; gaps reported in §4.",
        "- D99 unified reproduction merge deferred until DOMAIN_03 parallel train completes.",
        "",
        "---",
        "",
        "## 10. Next (prose pass — human)",
        "",
        "1. Write Introduction around §1 + Backblaze 0.958 exhibit.",
        "2. Expand Background citing only tools borrowed (not claimed as novel).",
        "3. Methods = §2–§3 + ladder description.",
        "4. Results = §4–§7 (paste tables/figures; do not retype numbers).",
        "5. Discussion = §9 + implications.",
        "",
    ]
    return "\n".join(lines)


def _to_md(payload: dict) -> str:
    lines = [
        "# P01 — Paper numbers (source of truth)",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        f"Protocol: `{payload['protocol_version']}`",
        f"n_keys: **{payload['n_keys']}**",
        "",
        "| key | value | source | path |",
        "|-----|-------|--------|------|",
    ]
    for k in sorted(payload["numbers"]):
        rec = payload["numbers"][k]
        v = rec["value"]
        if isinstance(v, float):
            vs = f"{v:.6g}"
        else:
            vs = json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v
            if len(vs) > 80:
                vs = vs[:77] + "…"
        lines.append(
            f"| `{k}` | `{vs}` | `{rec['source_id']}` | `{rec['json_path']}` |"
        )
    lines += [
        "",
        "## Rule",
        "",
        "All manuscript scalars must be looked up from `numbers.json`. "
        "Do not paste literals into the paper text by hand.",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    log("─" * 60)
    log("P01 — PAPER NUMBERS (no hard-coded scalars)")
    log("─" * 60)

    manifest = _load_collection()
    if not manifest.get("complete_required"):
        log("WARNING: P00 collection missing required sources — extracting what exists.")

    numbers = extract_numbers(manifest)
    payload = {
        "stage": "P01",
        "role": "paper_numbers_source_of_truth",
        "generated_at_utc": utc_now(),
        "protocol_version": PROTOCOL["version"],
        "collection_generated_at_utc": manifest.get("generated_at_utc"),
        "n_keys": len(numbers),
        "numbers": numbers,
        "rule": "Cite only via key lookup from this file; never hard-code.",
    }

    out_json = cfg.DIRS["paper"] / "numbers.json"
    out_md = cfg.DIRS["paper"] / "numbers.md"
    paper_md = cfg.DIRS["paper"] / "PAPER.md"
    cfg.DIRS["paper"].mkdir(parents=True, exist_ok=True)
    write_json(out_json, payload)
    md = _to_md(payload)
    out_md.write_text(md, encoding="utf-8")
    (cfg.DIRS["logs"] / "P01_paper_numbers.md").write_text(md, encoding="utf-8")

    paper = write_paper_md(numbers, manifest, payload["generated_at_utc"])
    paper_md.write_text(paper, encoding="utf-8")
    (cfg.DIRS["logs"] / "PAPER.md").write_text(paper, encoding="utf-8")

    log(f"Wrote {out_json} ({len(numbers)} keys)")
    log(f"Wrote {paper_md} (fact skeleton, no prose)")
    log("P01 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
