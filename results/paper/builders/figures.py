"""Figure builders F02–F05 (+ F01 Graphviz compile). Artefatos.md §§10–13."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
STYLE = ROOT / "results" / "paper" / "style"
FIG = ROOT / "results" / "paper" / "figures"
sys.path.insert(0, str(STYLE))

import colors as C  # noqa: E402
from mpl_setup import apply_paper_style, figsize, save_pdf  # noqa: E402
from numbers_io import fmt_float, fmt_p, load_numbers, v  # noqa: E402


def build_f01() -> Path:
    """Compile Graphviz workflow → PDF (artefatos.md §1), max quality."""
    gv = FIG / "F01_workflow.gv"
    pdf = FIG / "F01_workflow.pdf"
    if not gv.exists():
        raise FileNotFoundError(gv)
    # Prefer cairo PDF (crisper text); fall back to default PDF renderer.
    for fmt in ("pdf:cairo", "pdf"):
        try:
            subprocess.run(
                [
                    "dot",
                    f"-T{fmt}",
                    "-Gdpi=600",
                    "-Ndpi=600",
                    "-Edpi=600",
                    str(gv),
                    "-o",
                    str(pdf),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            return pdf
        except subprocess.CalledProcessError:
            continue
    raise RuntimeError(f"Graphviz failed to compile {gv}")


def build_f02(numbers: dict[str, Any] | None = None) -> Path:
    """F02_h1_c_vs_ibs — slopegraph of C vs IBS ranks by domain."""
    import matplotlib.pyplot as plt

    apply_paper_style()
    n = numbers or load_numbers()

    panels = [
        (
            "D1",
            C.DOMAIN["D1"],
            C.DOMAIN_LABEL["D1"],
            [
                ("cox_ablated", float(v(n, "H1.DOMAIN_01.cox_ablated.C")), float(v(n, "H1.DOMAIN_01.cox_ablated.IBS"))),
                ("cox_full", float(v(n, "H1.DOMAIN_01.cox_full.C")), float(v(n, "H1.DOMAIN_01.cox_full.IBS"))),
            ],
        ),
        (
            "D2",
            C.DOMAIN["D2"],
            C.DOMAIN_LABEL["D2"],
            [
                ("cox_classical", float(v(n, "H1.DOMAIN_02.cox_classical.C")), float(v(n, "H1.DOMAIN_02.cox_classical.IBS"))),
                ("cox_xgboost", float(v(n, "H1.DOMAIN_02.cox_xgboost.C")), float(v(n, "H1.DOMAIN_02.cox_xgboost.IBS"))),
            ],
        ),
        (
            "D3",
            C.DOMAIN["D3"],
            C.DOMAIN_LABEL["D3"],
            [
                ("rsf_behavioural", float(v(n, "H1.DOMAIN_03.rsf_behavioural.C")), float(v(n, "H1.DOMAIN_03.rsf_behavioural.IBS"))),
                ("rsf_combined", float(v(n, "H1.DOMAIN_03.rsf_combined.C")), float(v(n, "H1.DOMAIN_03.rsf_combined.IBS"))),
                ("rsf_content", float(v(n, "H1.DOMAIN_03.rsf_content.C")), float(v(n, "H1.DOMAIN_03.rsf_content.IBS"))),
            ],
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=figsize(7.0, 3.2))
    for ax, (dom, color, title, models) in zip(axes, panels):
        ax.patch.set_facecolor(color)
        ax.patch.set_alpha(0.05)
        c_vals = [m[1] for m in models]
        ibs_vals = [m[2] for m in models]
        n_m = len(models)
        c_rank = _rank_high_good(c_vals)  # larger C → higher rank (top)
        ibs_rank = _rank_low_good(ibs_vals)  # smaller IBS → higher rank

        x_c, x_ibs = 0.0, 1.0
        line_color = C.REJECT if dom == "D2" else C.SECONDARY
        for i, (name, _, _) in enumerate(models):
            y0, y1 = c_rank[i], ibs_rank[i]
            ax.plot([x_c, x_ibs], [y0, y1], color=line_color, linewidth=1.4, zorder=1)
            ax.scatter([x_c], [y0], s=48, color=color, marker="o", zorder=2, edgecolors=color)
            ax.scatter(
                [x_ibs],
                [y1],
                s=48,
                facecolors="none",
                edgecolors=color,
                marker="o",
                linewidths=1.4,
                zorder=2,
            )
            # model label at mid if needed — use y tick labels via twin? put names on left
            ax.annotate(
                name,
                xy=(x_c, y0),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
                color=C.NULL,
            )

        ax.set_xlim(-0.15, 1.15)
        ax.set_ylim(0.5, n_m + 0.95)
        ax.set_yticks(range(1, n_m + 1))
        ax.set_yticklabels([str(i) for i in range(1, n_m + 1)], fontsize=10)
        ax.set_xticks([x_c, x_ibs])
        ax.set_xticklabels(["C-index", "IPCW-IBS"], fontsize=11)
        ax.tick_params(axis="both", labelsize=10)
        ax.set_ylabel("Rank (higher = better)" if dom == "D1" else "", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.grid(False)

    # Shared legend
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C.NULL, markersize=9, label="C-index rank"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="none",
            markeredgecolor=C.NULL,
            markersize=9,
            label="IPCW-IBS rank",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
        fontsize=11,
    )
    fig.tight_layout()
    out = FIG / "F02_h1_c_vs_ibs.pdf"
    save_pdf(fig, out)
    plt.close(fig)
    return out


def build_f03(numbers: dict[str, Any] | None = None) -> Path:
    """F03_h2_dcal_backblaze — D-cal histogram for D1 cox_full."""
    import matplotlib.pyplot as plt

    apply_paper_style()
    n = numbers or load_numbers()
    props = [float(x) for x in v(n, "ladder.d01.cox_full.dcal_observed_proportions")]
    c_val = float(v(n, "ladder.d01.cox_full.C"))
    p_val = float(v(n, "ladder.d01.cox_full.dcal_p"))
    n_bins = int(v(n, "ladder.d01.cox_full.dcal_n_bins"))
    expected = 1.0 / n_bins

    fig, ax = plt.subplots(figsize=figsize(5.2, 3.8))
    x = np.arange(1, n_bins + 1)
    ax.bar(
        x,
        props,
        color=C.DOMAIN["D1"],
        alpha=0.85,
        edgecolor=C.EDGE,
        linewidth=0.5,
        width=0.85,
    )
    ax.axhline(expected, color=C.NULL, linestyle="--", linewidth=1.4)
    ax.text(
        n_bins + 0.35,
        expected,
        "Expected under calibration",
        color=C.NULL,
        fontsize=11,
        va="bottom",
        ha="right",
    )
    ax.set_xlabel("Predicted survival probability quantile bin", fontsize=13)
    ax.set_ylabel("Observed proportion", fontsize=13)
    ax.set_xticks(x)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_ylim(0, max(props) * 1.18)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}"))

    ann = f"C = {fmt_float(c_val, 3)}\nD-Cal p = {_sci_unicode(p_val)}"
    ax.text(
        0.97,
        0.97,
        ann,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=12,
        color=C.NULL,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=C.SECONDARY, linewidth=0.6),
    )
    fig.tight_layout()
    out = FIG / "F03_h2_dcal_backblaze.pdf"
    save_pdf(fig, out)
    plt.close(fig)
    return out


def build_f04(numbers: dict[str, Any] | None = None) -> Path:
    """F04_h3_competing_risks — point-range by rating stratum."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    apply_paper_style()
    n = numbers or load_numbers()
    order = ["AA", "A", "B", "C", "D", "E", "F"]
    deltas, lows, highs, supports = [], [], [], []
    for rating in order:
        pref = f"H3.stratum.{rating}"
        d = float(v(n, f"{pref}.delta"))
        ci = v(n, f"{pref}.ci")
        deltas.append(d)
        lows.append(float(ci[0]))
        highs.append(float(ci[1]))
        supports.append(bool(v(n, f"{pref}.supports")))

    threshold = float(v(n, "protocol.H3.delta_threshold"))
    fig, ax = plt.subplots(figsize=figsize(5.2, 4.0))
    x = np.arange(len(order))
    # trend guide
    ax.plot(x, deltas, color=C.SECONDARY, alpha=0.5, linewidth=1.2, zorder=1)
    for i, (d, lo, hi, sup) in enumerate(zip(deltas, lows, highs, supports)):
        color = C.REJECT if sup else C.DOMAIN["D2"]
        ax.errorbar(
            i,
            d,
            yerr=[[d - lo], [hi - d]],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=4,
            markersize=8,
            zorder=2,
        )
    ax.axhline(0.0, color=C.NULL, linestyle="-", linewidth=0.8)
    ax.axhline(threshold, color=C.SECONDARY, linestyle="--", linewidth=1.2)
    ax.text(
        len(order) - 0.05,
        threshold,
        rf"Pre-registered threshold ($\delta = {threshold:g}$)",
        fontsize=11,
        color=C.SECONDARY,
        ha="right",
        va="bottom",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(order, fontsize=12)
    ax.set_xlabel("Credit rating", fontsize=13)
    ax.set_ylabel("Bias in 12-month default probability\n(naive − Aalen–Johansen)", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}"))

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C.REJECT, markersize=9, label="Supports H3 (D, E, F)"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=C.DOMAIN["D2"],
            markersize=9,
            label="Does not support (AA–C)",
        ),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=1,
        fontsize=12,
    )
    fig.tight_layout()
    out = FIG / "F04_h3_competing_risks.pdf"
    save_pdf(fig, out)
    plt.close(fig)
    return out


def build_f05(numbers: dict[str, Any] | None = None) -> Path:
    """F05_h5_brier_strip — horizon Brier for D3 models."""
    import matplotlib.pyplot as plt

    apply_paper_style()
    n = numbers or load_numbers()
    horizons = [int(h) for h in v(n, "protocol.H5.horizons_months")]
    band = v(n, "protocol.H5.global_cindex_band")
    band_lo, band_hi = float(band[0]), float(band[1])

    models = [
        "rsf_behavioural",
        "rsf_content",
        "rsf_combined",
    ]
    primary = str(v(n, "H5.primary_model_id"))

    fig, ax = plt.subplots(figsize=figsize(5.5, 4.2))
    x = np.arange(len(horizons))
    for mid in models:
        brier = v(n, f"H5.{mid}.brier")
        ys = [float(brier[str(h)]) for h in horizons]
        c_val = float(v(n, f"H5.{mid}.C"))
        in_band = band_lo <= c_val <= band_hi
        is_primary = mid == primary
        if is_primary and in_band:
            ax.plot(
                x,
                ys,
                color=C.DOMAIN["D3"],
                linestyle="-",
                linewidth=2.4,
                marker="o",
                markersize=8,
                markerfacecolor=C.DOMAIN["D3"],
                label=f"{mid} (C = {fmt_float(c_val, 3)}, in band)",
            )
        else:
            status = "in band" if in_band else "out of band"
            ax.plot(
                x,
                ys,
                color=C.DOMAIN["D3"],
                linestyle="--",
                linewidth=1.6,
                alpha=0.55,
                marker="o",
                markersize=8,
                markerfacecolor="none",
                markeredgecolor=C.DOMAIN["D3"],
                label=f"{mid} (C = {fmt_float(c_val, 3)}, {status})",
            )

    ax.set_xticks(x)
    ax.set_xticklabels([str(h) for h in horizons], fontsize=12)
    ax.set_xlabel("Scoring horizon (months)", fontsize=13)
    ax.set_ylabel("IPCW Brier score", fontsize=13)
    ax.tick_params(axis="both", labelsize=11)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.3f}"))
    ax.legend(
        frameon=False,
        fontsize=11,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=1,
    )
    fig.tight_layout()
    out = FIG / "F05_h5_brier_strip.pdf"
    save_pdf(fig, out)
    plt.close(fig)
    return out


def build_f06(sweep: dict[str, Any] | None = None) -> Path:
    """F06_copula_sweep_d1 — CG-IPCW Uno C and IBS vs Kendall τ (D1 cox_full)."""
    import json

    import matplotlib.pyplot as plt

    apply_paper_style()
    if sweep is None:
        path = ROOT / "results" / "paper" / "numbers_copula_sweep_d1.json"
        sweep = json.loads(path.read_text())

    tau = np.asarray(sweep["tau_grid"], dtype=float)
    c_mean = np.asarray(sweep["uno_c_adjusted"]["mean"], dtype=float)
    c_lo = np.asarray(sweep["uno_c_adjusted"]["ci_lo"], dtype=float)
    c_hi = np.asarray(sweep["uno_c_adjusted"]["ci_hi"], dtype=float)
    ibs_mean = np.asarray(sweep["ibs_adjusted"]["mean"], dtype=float)
    ibs_lo = np.asarray(sweep["ibs_adjusted"]["ci_lo"], dtype=float)
    ibs_hi = np.asarray(sweep["ibs_adjusted"]["ci_hi"], dtype=float)

    ref = sweep.get("reference") or {}
    uno0 = float(ref.get("uno_cg_tau0", c_mean[0]))
    ibs0 = float(ref.get("ibs_cg_tau0", ibs_mean[0]))
    harrell = float(ref.get("harrell_unadjusted", 0.9595))
    color = C.DOMAIN["D1"]

    fig, axes = plt.subplots(1, 2, figsize=figsize(7.0, 2.8))

    # Left: Uno C (CG-IPCW)
    ax = axes[0]
    ax.fill_between(tau, c_lo, c_hi, color=color, alpha=0.15, linewidth=0)
    ax.plot(tau, c_mean, color=color, linewidth=1.5)
    ax.axvline(0.0, color=C.NULL, linestyle="--", linewidth=1.0)
    ax.axhline(uno0, color=C.SECONDARY, linestyle="--", linewidth=1.0)
    ax.text(
        0.02,
        0.02,
        "Independent censoring\n(standard)",
        transform=ax.transAxes,
        color=C.NULL,
        fontsize=8,
        va="bottom",
    )
    ax.text(
        0.98,
        0.98,
        f"Unadjusted Uno C = {uno0:.3f}",
        transform=ax.transAxes,
        color=C.SECONDARY,
        fontsize=8,
        ha="right",
        va="top",
    )
    ax.set_xlabel("Assumed censoring dependence (Kendall's $\\tau$)")
    ax.set_ylabel("Adjusted Uno C (CG-IPCW)")
    ax.set_xlim(0.0, 0.75)
    pad = 0.02 * max(c_hi.max() - c_lo.min(), 1e-3)
    ax.set_ylim(c_lo.min() - pad, max(c_hi.max(), harrell) + pad)

    # Right: IBS (CG-IPCW)
    ax = axes[1]
    ax.fill_between(tau, ibs_lo, ibs_hi, color=color, alpha=0.15, linewidth=0)
    ax.plot(tau, ibs_mean, color=color, linewidth=1.5)
    ax.axvline(0.0, color=C.NULL, linestyle="--", linewidth=1.0)
    ax.axhline(ibs0, color=C.SECONDARY, linestyle="--", linewidth=1.0)
    ax.text(
        0.02,
        0.02,
        "Independent censoring\n(standard)",
        transform=ax.transAxes,
        color=C.NULL,
        fontsize=8,
        va="bottom",
    )
    ax.text(
        0.98,
        0.98,
        f"Unadjusted IBS = {ibs0:.3f}",
        transform=ax.transAxes,
        color=C.SECONDARY,
        fontsize=8,
        ha="right",
        va="top",
    )
    ax.set_xlabel("Assumed censoring dependence (Kendall's $\\tau$)")
    ax.set_ylabel("Adjusted IPCW-IBS (CG $\\hat{G}$)")
    ax.set_xlim(0.0, 0.75)
    pad = 0.02 * max(ibs_hi.max() - ibs_lo.min(), 1e-3)
    ax.set_ylim(ibs_lo.min() - pad, ibs_hi.max() + pad)

    fig.tight_layout()
    out = FIG / "F06_copula_sweep_d1.pdf"
    save_pdf(fig, out)
    plt.close(fig)
    return out


def _rank_high_good(values: list[float]) -> list[float]:
    """Larger value → higher rank number (best at top of y-axis)."""
    n = len(values)
    order_asc = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    for pos, i in enumerate(order_asc):
        ranks[i] = float(pos + 1)  # smallest → 1, largest → n
    return ranks


def _rank_low_good(values: list[float]) -> list[float]:
    """Smaller value → higher rank number (best at top of y-axis)."""
    n = len(values)
    order_asc = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    for pos, i in enumerate(order_asc):
        ranks[i] = float(n - pos)  # smallest → n
    return ranks


def _sci_unicode(p: float) -> str:
    s = f"{float(p):.2e}"
    mant, exp = s.split("e")
    return f"{mant} × 10^{int(exp)}"


BUILDERS: dict[str, Callable[..., Path]] = {
    "F01": build_f01,
    "F02": build_f02,
    "F03": build_f03,
    "F04": build_f04,
    "F05": build_f05,
    "F06": build_f06,
}


def build_all(numbers: dict[str, Any] | None = None) -> list[tuple[str, Path]]:
    n = numbers or load_numbers()
    out = [("F01", build_f01())]
    for name in ("F02", "F03", "F04", "F05"):
        out.append((name, BUILDERS[name](n)))
    # F06 reads its own JSON (optional if present)
    f06_json = ROOT / "results" / "paper" / "numbers_copula_sweep_d1.json"
    if f06_json.exists():
        out.append(("F06", build_f06()))
    return out
