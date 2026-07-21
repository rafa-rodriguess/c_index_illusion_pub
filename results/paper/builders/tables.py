"""Table builders T01–T08 (artefatos.md). All scalars from numbers.json."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[3]
STYLE = ROOT / "results" / "paper" / "style"
sys.path.insert(0, str(STYLE))

from booktabs import write_table  # noqa: E402
from numbers_io import (  # noqa: E402
    fmt_float,
    fmt_int,
    fmt_p,
    fmt_p_tex,
    fmt_signed_tex,
    load_numbers,
    v,
    yes_no,
)

TABLES = ROOT / "results" / "paper" / "tables"

# Protocol freeze §3.3 reproduction tiers — loaded from numbers.json at build time
def _tier(abs_gap: float, numbers: dict[str, Any]) -> str:
    strict = float(v(numbers, "protocol.repro.tier_strict"))
    approx = float(v(numbers, "protocol.repro.tier_approx"))
    if abs_gap <= strict:
        return "Strict"
    if abs_gap <= approx:
        return "Approximate"
    return "Fail"


def build_t01(numbers: dict[str, Any] | None = None) -> tuple[Path, Path]:
    """T01_domains_baselines — descriptive sampling frame."""
    n = numbers or load_numbers()
    d1_c = fmt_float(v(n, "repro.DOMAIN_01.harrell_cindex.paper"), 3)
    # D3 published band from the 18 paper cells
    papers = [
        float(v(n, k))
        for k in sorted(n)
        if k.startswith("repro.DOMAIN_03.cindex_") and k.endswith(".paper")
    ]
    d3_lo, d3_hi = min(papers), max(papers)
    d3_band = f"{d3_lo:.2f}–{d3_hi:.2f} (range across {len(papers)} cells)"

    columns = [
        "Domain",
        "Baseline (Author, Year)",
        "Dataset",
        "Event",
        "Reported C-index",
        "Public",
    ]
    rows = [
        [
            "D1: Engineering Reliability",
            "Ahmed \\& Green (2024)",
            "Backblaze Drive Stats (SMART)",
            "Hard-drive failure",
            d1_c,
            r"\checkmark",
        ],
        [
            "D2: Structured Finance",
            "Bone-Winkel \\& Reichenbach (2024)",
            "Bondora P2P Loans",
            "Loan default",
            "--- (rating-based, no single scalar)",
            r"\checkmark",
        ],
        [
            "D3: Digital Platforms",
            "Abedi Firouzjaei (2022)",
            "Stack Exchange Data Dump",
            "User disengagement",
            d3_band,
            r"\checkmark",
        ],
    ]
    return write_table(
        TABLES / "T01_domains_baselines",
        columns,
        rows,
        caption=(
            "The three domains audited in this study. All three datasets are public; "
            "selection criteria are described in §3.2."
        ),
    )


def build_t02(numbers: dict[str, Any] | None = None) -> tuple[Path, Path]:
    """T02_anchor_oracle — synthetic Table 2 reproduction."""
    n = numbers or load_numbers()
    scenarios = [
        ("Random", "random.random"),
        ("Independent", "independent.independent"),
        ("Dependent ($\\tau=0.25$)", "dep_tau25.dep_tau25"),
        ("Dependent ($\\tau=0.50$)", "dep_tau50.dep_tau50"),
        ("Dependent ($\\tau=0.75$)", "dep_tau75.dep_tau75"),
    ]
    columns = [
        "Censoring Scenario",
        "$C_{\\mathrm{oracle}}$ (ours)",
        "$C_{\\mathrm{oracle}}$ (paper)",
        "Gap",
        "$\\mathrm{IBS}_{\\mathrm{oracle}}$ (ours)",
        "$\\mathrm{IBS}_{\\mathrm{oracle}}$ (paper)",
        "Gap",
    ]
    rows = []
    c_gaps = []
    ibs_gaps = []
    for label, key in scenarios:
        pref = f"anchor.table2.{key}"
        c_o = float(v(n, f"{pref}.ci_oracle_ours"))
        c_p = float(v(n, f"{pref}.ci_oracle_paper"))
        c_g = float(v(n, f"{pref}.ci_oracle_gap"))
        i_o = float(v(n, f"{pref}.ibs_oracle_ours"))
        i_p = float(v(n, f"{pref}.ibs_oracle_paper"))
        i_g = float(v(n, f"{pref}.ibs_oracle_gap"))
        c_gaps.append(abs(c_g))
        ibs_gaps.append(abs(i_g))
        rows.append(
            [
                label,
                fmt_float(c_o, 4),
                fmt_float(c_p, 4),
                fmt_signed_tex(c_g, 4),
                fmt_float(i_o, 4),
                fmt_float(i_p, 4),
                fmt_signed_tex(i_g, 4),
            ]
        )
    i_max_c = max(range(len(c_gaps)), key=lambda i: c_gaps[i])
    i_max_i = max(range(len(ibs_gaps)), key=lambda i: ibs_gaps[i])
    bold = [[False] * 7 for _ in rows]
    for i in range(len(rows)):
        bold[i][3] = i == i_max_c
        bold[i][6] = i == i_max_i

    n_seeds = int(v(n, "anchor.table2.n_seeds"))
    note = (
        f"{n_seeds} random seeds. Maximum absolute gap: "
        f"{max(c_gaps):.4f} (discrimination), {max(ibs_gaps):.4f} (proper score)."
    )
    return write_table(
        TABLES / "T02_anchor_oracle",
        columns,
        rows,
        bold_mask=bold,
        table_note=note,
        caption=(
            "Reproduction of the anchor paper's Table 2 (oracle discrimination and proper "
            "score under five censoring regimes) inside our pipeline. Oracle metrics use "
            "the true, simulation-known event times and are only available in this "
            "synthetic setting. See §3.4."
        ),
    )


def build_t03(numbers: dict[str, Any] | None = None) -> tuple[Path, Path]:
    """T03_hypotheses_summary — pre-registered rules (thresholds from protocol)."""
    n = numbers or load_numbers()
    h2_c = fmt_float(float(v(n, "protocol.H2.c_h_threshold")), 2)
    h3_d = fmt_float(float(v(n, "protocol.H3.delta_threshold")), 2)
    h3_h = int(v(n, "protocol.H3.horizon_months"))
    h3_k = int(v(n, "protocol.H3.min_strata_consistent"))
    h4_d = fmt_float(float(v(n, "protocol.H4.delta_c_threshold")), 2)
    alpha = fmt_float(float(v(n, "protocol.globals.alpha")), 2)
    tau_ceil = fmt_float(float(v(n, "protocol.H1.tau_reject_ceiling")), 1)
    columns = ["\\#", "Hypothesis", "Domain", "Test Statistic", "Rejection Threshold"]
    rows = [
        [
            "H1",
            "Ranking inversion",
            "All (§3.8/C00.2)",
            "$\\tau_K$ (C-index rank vs.\\ IPCW-IBS rank)",
            f"$\\tau_K \\leq {tau_ceil}$ and bootstrap $p < {alpha}$ (C00.1)",
        ],
        [
            "H2",
            "Discrimination–calibration dissociation",
            "D1",
            "D-Calibration $p$-value",
            f"$C_H \\geq {h2_c}$ and $p < {alpha}$",
        ],
        [
            "H3",
            "Competing-risks bias",
            "D2",
            f"$|F_{{\\mathrm{{naive}}}} - F_{{\\mathrm{{AJ}}}}|$ @ {h3_h}mo",
            f"$> {h3_d}$ and CI excludes 0 and $\\geq {h3_k}$ supporting strata$^{{\\dagger}}$",
        ],
        [
            "H4",
            "Feature-concentration ablation",
            "D1",
            "$\\exists$ ablation set with $\\Delta C_H$",
            f"$\\geq {h4_d}$ and CI non-overlapping",
        ],
        [
            "H5",
            "Horizon-masked degradation",
            "D3",
            "Brier(36mo) $-$ Brier(12mo)",
            "Monotonic increase, $> 2$ combined SE, and $C$ in reported band",
        ],
        [
            "H\\_meta",
            "C-index-only proxy validity",
            "---",
            "Count of Holm-rejected primary hypotheses",
            "$\\geq 3$ of 5",
        ],
    ]
    note = (
        r"$^{\dagger}$C00 freeze text: ``$\geq 3$ of 5 rating strata'' (expected count at "
        r"freeze). Bondora has seven ratings (AA--F); operative threshold is absolute "
        r"\texttt{min\_strata\_consistent}$="
        f"{h3_k}"
        r". See Appendix A."
    )
    return write_table(
        TABLES / "T03_hypotheses_summary",
        columns,
        rows,
        preamble=r"% Use \footnotesize around \input{...} if the table is too wide.",
        col_spec=r"lp{1.8cm}p{2.2cm}p{3.6cm}p{4.2cm}",
        table_note=note,
        caption=(
            "The five pre-registered hypotheses and the meta-hypothesis, with test "
            "statistics and decision rules fixed in the protocol freeze (Appendix A) "
            "before any test data were examined."
        ),
    )


def build_t04(numbers: dict[str, Any] | None = None) -> tuple[Path, Path]:
    """T04_reproduction_headline."""
    n = numbers or load_numbers()
    rows_spec = [
        (
            "D1",
            "Cox",
            float(v(n, "repro.DOMAIN_01.harrell_cindex.paper")),
            float(v(n, "repro.DOMAIN_01.harrell_cindex.ours")),
            float(v(n, "repro.DOMAIN_01.harrell_cindex.gap")),
            False,
        ),
        (
            "D2",
            "Cox (linear)",
            float(v(n, "repro.DOMAIN_02.cindex_linear_test.paper")),
            float(v(n, "repro.DOMAIN_02.cindex_linear_test.ours")),
            float(v(n, "repro.DOMAIN_02.cindex_linear_test.gap")),
            False,
        ),
        (
            "D2",
            "Cox (boosted)",
            float(v(n, "repro.DOMAIN_02.cindex_boosted_test.paper")),
            float(v(n, "repro.DOMAIN_02.cindex_boosted_test.ours")),
            float(v(n, "repro.DOMAIN_02.cindex_boosted_test.gap")),
            True,  # approximate row — light shade
        ),
    ]
    d3_mean = float(v(n, "repro.DOMAIN_03.cindex_mean_abs_gap"))
    d3_n = int(v(n, "repro.DOMAIN_03.cindex_n_cells"))
    papers = [
        float(v(n, k))
        for k in sorted(n)
        if k.startswith("repro.DOMAIN_03.cindex_") and k.endswith(".paper")
    ]
    d3_band = f"{min(papers):.2f}–{max(papers):.2f}"

    columns = ["Domain", "Model", "C (reported)", "C (reproduced)", "Gap", "Tier"]
    rows = []
    bold = []
    colors: list[str | None] = []
    for dom, model, paper, ours, gap, shade in rows_spec:
        tier = _tier(abs(gap), n)
        rows.append(
            [
                dom,
                model,
                fmt_float(paper, 4),
                fmt_float(ours, 4),
                fmt_signed_tex(gap, 4),
                tier,
            ]
        )
        bold.append([False, False, False, False, False, tier == "Strict"])
        colors.append("gray!10" if shade else None)

    rows.append(
        [
            "D3",
            f"RSF ({d3_n}-cell mean)",
            d3_band,
            d3_band,
            "n.a.",
            _tier(d3_mean, n),
        ]
    )
    bold.append([False, False, False, False, False, _tier(d3_mean, n) == "Strict"])
    colors.append(None)

    strict = fmt_float(float(v(n, "protocol.repro.tier_strict")), 2)
    approx = fmt_float(float(v(n, "protocol.repro.tier_approx")), 2)
    d1_hist = fmt_float(float(v(n, "protocol.repro.d1_unfiltered_c_cited")), 3)
    note = (
        f"Tiers per the reproduction criterion of §3.3: strict $\\leq {strict}$, "
        f"approximate $({strict}, {approx}]$. Domain 3 reports the published and "
        f"reproduced cell-wise bands; mean absolute gap across all {d3_n} cells is "
        f"${fmt_float(d3_mean, 4)}$ (strict). D1's earlier iteration (unfiltered cohort) "
        f"reproduced $C = {d1_hist}$; see §4.1 and §5.2."
    )
    return write_table(
        TABLES / "T04_reproduction_headline",
        columns,
        rows,
        bold_mask=bold,
        row_color=colors,
        table_note=note,
        caption=(
            "Reproduction gaps for each baseline's headline discrimination quantity. "
            "Full per-domain reproduction detail — including every under-specified "
            "preprocessing choice — is in Appendix B; the complete 18-cell Domain 3 "
            "table is in Appendix D."
        ),
    )


def build_t05(numbers: dict[str, Any] | None = None) -> tuple[Path, Path]:
    """T05_h1_tau_by_domain."""
    n = numbers or load_numbers()
    columns = ["Domain", "$\\tau_K$", "Bootstrap 95\\% CI", "Reject (C00.1)"]
    rows = []
    italic = []
    for dom, label in (("DOMAIN_01", "D1"), ("DOMAIN_02", "D2"), ("DOMAIN_03", "D3")):
        tau = float(v(n, f"H1.{dom}.tau_K"))
        ci = v(n, f"H1.{dom}.boot_ci")
        reject = bool(v(n, f"H1.{dom}.reject"))
        tau_s = fmt_float(tau, 3) if tau >= 0 else f"$-{fmt_float(abs(tau), 3)}$"
        lo, hi = float(ci[0]), float(ci[1])
        lo_s = fmt_float(lo, 3) if lo >= 0 else f"$-{fmt_float(abs(lo), 3)}$"
        hi_s = fmt_float(hi, 3) if hi >= 0 else f"$-{fmt_float(abs(hi), 3)}$"
        rows.append([label, tau_s, f"[{lo_s}, {hi_s}]", yes_no(reject)])
        # Italics for D2 τ (qualitatively striking, not significant)
        italic.append([False, label == "D2", False, False])
    return write_table(
        TABLES / "T05_h1_tau_by_domain",
        columns,
        rows,
        italic_mask=italic,
        caption=(
            "Kendall's $\\tau$ between the C-index ranking and the "
            "IPCW-Integrated-Brier-Score ranking, within each domain. "
            "$\\tau = -1$ in Domain 2 indicates complete rank inversion between the "
            "two metrics; the wide confidence interval reflects the two-model "
            "comparison's lack of statistical power (§5.3). See Figure 2."
        ),
    )


def build_t06(numbers: dict[str, Any] | None = None) -> tuple[Path, Path]:
    """T06_dcalibration_all_models (+ post-hoc ECE / max bin deviation)."""
    n = numbers or load_numbers()
    order = [
        ("D1", "d01", "cox_full"),
        ("D1", "d01", "cox_ablated"),
        ("D2", "d02", "cox_classical"),
        ("D2", "d02", "cox_xgboost"),
        ("D3", "d03", "rsf_behavioural"),
        ("D3", "d03", "rsf_content"),
        ("D3", "d03", "rsf_combined"),
    ]
    columns = [
        "Domain",
        "Model",
        "C-index",
        "D-Cal $p$-value",
        "ECE (pp)",
        "Max bin dev (pp)",
        "Reject ($\\alpha=0.05$)",
    ]
    rows = []
    bold = []
    for dom, slug, mid in order:
        c = float(v(n, f"ladder.{slug}.{mid}.C"))
        p = float(v(n, f"ladder.{slug}.{mid}.dcal_p"))
        rej = bool(v(n, f"ladder.{slug}.{mid}.dcal_reject"))
        ece_pp = float(v(n, f"ladder.{slug}.{mid}.dcal_ece_pp"))
        max_pp = float(v(n, f"ladder.{slug}.{mid}.dcal_max_bin_dev_pp"))
        rows.append(
            [
                dom,
                f"\\texttt{{{mid}}}",
                fmt_float(c, 4),
                fmt_p_tex(p),
                fmt_float(ece_pp, 2),
                fmt_float(max_pp, 2),
                yes_no(rej),
            ]
        )
        is_h2 = mid == "cox_full"
        bold.append([is_h2] * 7)
    note = (
        "All seven models reject D-Calibration; H2 formally targets the D1 "
        "\\texttt{cox\\_full} row (bold), the only model with "
        f"$C_H \\geq {fmt_float(float(v(n, 'protocol.H2.c_h_threshold')), 2)}$ "
        "per the pre-registered rule. ECE = $\\sum_i (n_i/N)|\\hat p_i - 0.10|$ "
        "from the ten D-Calibration bins (percentage points); max bin deviation "
        "is $\\max_i |\\hat p_i - 0.10|$."
    )
    return write_table(
        TABLES / "T06_dcalibration_all_models",
        columns,
        rows,
        bold_mask=bold,
        table_note=note,
        caption=(
            "D-Calibration test results for every reproduced model across all three "
            "domains, with post-hoc Expected Calibration Error (ECE) and maximum "
            "decile deviation. See Figure 3 for the calibration histogram of the bolded model."
        ),
    )


def build_t07(numbers: dict[str, Any] | None = None) -> tuple[Path, Path]:
    """T07_h3_by_stratum — safest → riskiest."""
    n = numbers or load_numbers()
    order = ["AA", "A", "B", "C", "D", "E", "F"]
    columns = ["Rating", "$n$", "$\\Delta$ (naive $-$ AJ)", "95\\% CI", "Supports H3"]
    rows = []
    bold = []
    for rating in order:
        pref = f"H3.stratum.{rating}"
        delta = float(v(n, f"{pref}.delta"))
        ci = v(n, f"{pref}.ci")
        nn = int(v(n, f"{pref}.n"))
        supports = bool(v(n, f"{pref}.supports"))
        rows.append(
            [
                rating,
                fmt_int(nn),
                fmt_float(delta, 4),
                f"[{fmt_float(ci[0], 4)}, {fmt_float(ci[1], 4)}]",
                yes_no(supports),
            ]
        )
        bold.append([supports] * 5)
    return write_table(
        TABLES / "T07_h3_by_stratum",
        columns,
        rows,
        bold_mask=bold,
        caption=(
            "Bias in twelve-month cumulative incidence of default (naive estimator minus "
            "Aalen–Johansen), by Bondora credit rating. Bias grows monotonically with "
            "credit risk; only the three riskiest strata (D, E, F, bold) meet the "
            "pre-registered support threshold. See Figure 4."
        ),
    )


def build_t08(numbers: dict[str, Any] | None = None) -> tuple[Path, Path]:
    """T08_holm_verdict — money table."""
    n = numbers or load_numbers()
    hyps = [
        ("H1", "H1"),
        ("H2", "H2"),
        ("H3", "H3"),
        ("H4", "H4"),
        ("H5", "H5"),
    ]
    columns = ["Hypothesis", "Raw $p$", "Holm-adjusted $p$", "Reject $H_0$"]
    rows = []
    bold = []
    for label, key in hyps:
        raw = float(v(n, f"{key}.raw_p"))
        adj = float(v(n, f"{key}.adj_p"))
        rej = bool(v(n, f"{key}.holm_reject"))
        rows.append([label, fmt_p_tex(raw), fmt_p_tex(adj), yes_no(rej)])
        bold.append([rej] * 4)

    n_rej = int(v(n, "H_meta.n_holm_rejects"))
    meta_rej = bool(v(n, "H_meta.reject"))
    meta_cell = f"Rejected ({n_rej}/5)" if meta_rej else f"Not rejected ({n_rej}/5)"
    rows.append(["H\\_meta", "---", "---", meta_cell])
    bold.append([True, True, True, True])

    return write_table(
        TABLES / "T08_holm_verdict",
        columns,
        rows,
        bold_mask=bold,
        midrule_before=[5],
        row_color=[None, None, None, None, None, "gray!15"],
        caption=(
            "Final Holm-corrected family-wise verdict. Three of five pre-registered "
            "hypotheses reject; by the pre-registered rule ($\\geq 3$ of 5), the "
            "meta-hypothesis — that C-index-only evaluation is a reliable proxy outside "
            "healthcare — is rejected."
        ),
    )


BUILDERS: dict[str, Callable[..., tuple[Path, Path]]] = {
    "T01": build_t01,
    "T02": build_t02,
    "T03": build_t03,
    "T04": build_t04,
    "T05": build_t05,
    "T06": build_t06,
    "T07": build_t07,
    "T08": build_t08,
}


def build_all(numbers: dict[str, Any] | None = None) -> list[tuple[str, Path, Path]]:
    n = numbers or load_numbers()
    out = []
    for name, fn in BUILDERS.items():
        tex, csv = fn(n)
        out.append((name, tex, csv))
    return out
