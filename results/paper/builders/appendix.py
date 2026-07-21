"""Appendix builders A01–A04 (artefatos.md §15–18).

Tables/CSV live under ``results/paper/appendix/``. Scalars from
``numbers.json`` and ``results/reproduction/DOMAIN_0n_reproduction_table.json``.
Protocol text from ``results/harness/protocol_freeze.md`` + C00.4 / rejected
alternatives documented in ``roadmap.md`` / artefatos.
"""

from __future__ import annotations

import json
import re
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
    fmt_signed_tex,
    load_numbers,
    v,
)

APPENDIX = ROOT / "results" / "paper" / "appendix"
REPRO = ROOT / "results" / "reproduction"
PROTOCOL_MD = ROOT / "results" / "harness" / "protocol_freeze.md"

# ── Limitation rows + frase-guia (artefatos.md §16; roadmap L-tables) ─────────

LIMITATIONS: dict[str, list[list[str]]] = {
    "d01": [
        ["L1", "Author code inaccessible", "Footnote: GitLab gitlab.com/Jishan/deeplearning2023", "Repo private (see CODE_ACCESS.md)", "Best-effort reproduction from the published text"],
        ["L2", "Cohort ``>7 years'' (\\S4.1)", "12{,}993 healthy + 4{,}889 failed", "H6a: healthy calendar span $>7$y $\\cup$ all failed $\\rightarrow$ $\\sim$12{,}815 / 5{,}089", "Counts $\\pm\\sim$200; C matches; GitLab 404"],
        ["L3", "SMART snapshot for Cox", "\\S3.3.2/\\S7.1 silent on first/last/mean", "SMART raw on last observed day", "May shift C/HRs vs private pipeline"],
        ["L4", "SMART 190 $\\equiv$ 194", "HR reported for both", "Identical in this fleet $\\rightarrow$ drop 194", "Engineering deviation, not scientific"],
        ["L5", "L2 penalizer", "Names CoxPHFitter; silent on regularization", "penalizer$=$0.01 (Newton fails at 0.0)", "C/HRs depend mildly on this knob"],
        ["L6", "Cox hold-out", "\\S6 80/20 is for DeepNet; \\S7.1 C as GOF", "In-sample C on the fit", "Comparable to the paper's GOF framing"],
        ["L7", "15-day horizon", "\\S4.1/\\S6 labeling for DL classification", "Not applied to Cox", "Correct for the Cox target"],
    ],
    "d02": [
        ["L1", "No code repository", "Zenodo $=$ lifelines", "Best-effort from the published text", "Not bit-exact"],
        ["L2", "Public dump $\\neq$ private extract", "Retrieved 2024-01-03", "Kaggle + D00 align (as-of 2024-01-03; 36m; 2014--2020)", "Schema $\\sim$97 vs $\\sim$112 cols"],
        ["L3", "10-step preprocess", "\\S3.2 detailed", "src/domain2\\_preprocess.py", "Encoding may differ"],
        ["L4", "Repayments / IRR", "Table 1 IRR", "skipped", "Does not block H3 / Phase A"],
        ["L5", "XGB HPO", "Optuna + GPU", "Optuna TPE 40 trials (CPU hist)", "AA boosted $\\approx$ paper"],
        ["L6", "Investor xlsx", "---", "Not used", "Canonical $=$ LoanData.csv"],
        ["L7", "Bondora AA residual", "Table 1 0.1726", "Residual on full 2020 test definition", "Possible default/censoring definition mismatch"],
    ],
    "d03": [
        ["L1", "RSF backend", "PySurvival", "PySurvival 0.1.2 patched (d3-pysurvival)", "Aligned; fragile build on macOS"],
        ["L2", "Hyperparameters", "Grid $q$, $d$ not reported", "Notebook: 5 trees, depth 5, leaf 30", "Pickles do not reveal HPs"],
        ["L3", "C-index definition", "Utkin via PySurvival", "pysurvival.concordance\\_index", "Aligned"],
        ["L4", "Contributor filter", "\\S5.2 (Q$\\cup$A$\\cup$C$\\cup$U$\\cup$D)", "Applied (counts $>0$)", "Aligned to text"],
        ["L5", "CV seed", "notebook seed$=$None", "RANDOM\\_SEED + run\\_id", "Reproducible; mild deviation"],
    ],
}

FRASE_GUIA = {
    "d01": (
        "Even after matching the published population filter (ST4000DM000, 2013--2022, "
        "21 SMART raw features) and recovering a Cox C-index within 0.0015 of the headline "
        "0.958, several preprocessing choices required by the baseline remain under-specified "
        "and unverifiable because the linked GitLab repository is private. The residual gap "
        "is reported as a reproduction finding, not discarded as implementation error."
    ),
    "d02": (
        "After aligning a public LoanData dump to the authors' retrieve date and implementing "
        "\\S3.2-style preprocessing with Optuna-tuned XGB-Cox, the boosted AA default rate "
        "matches Table 1 within 0.001; residual Bondora-platform AA discrepancy and missing "
        "IRR remain documented reproduction findings rather than calendar artifacts."
    ),
    "d03": (
        "Using the author's user\\_features, \\S5.2 contributor filter, notebook CV "
        "(1\\% holdout + unshuffled 5-fold$\\times$30), and PySurvival RSF/concordance, "
        "we recover Table 8 C-indices within $\\sim$0.004 of the published means."
    ),
}

DOMAIN_TITLES = {
    "d01": "Domain 1 --- Backblaze / Ahmed \\& Green (2024)",
    "d02": "Domain 2 --- Bondora / Bone-Winkel \\& Reichenbach (2024)",
    "d03": "Domain 3 --- Stack Exchange / Abedi Firouzjaei (2022)",
}

COMMUNITY = {"p": "Politics", "ds": "Data Science", "cs": "Computer Science"}
FEATURE = {
    "behavioural": "Behavioural",
    "content": "Content",
    "combined": "Combined",
}


def _load_repro(domain: int) -> dict[str, Any]:
    path = REPRO / f"DOMAIN_0{domain}_reproduction_table.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _clean_source(src: Any) -> str:
    """Map internal audit notes to submission-ready source labels."""
    s = str(src or "").strip()
    replacements = {
        "ours (Appendix D publishes linear AA)": "This work",
        "ours / §3.6 KM": "This work",
        "paper §3.3": "Baseline §3.3",
        "paper Table 1 / §3.5": "Baseline Table 1 / §3.5",
        "paper §4.1": "Baseline §4.1",
        "paper Table 1 / footnote 14": "Baseline Table 1 / footnote 14",
        "paper Table 1": "Baseline Table 1",
        "paper Appendix D": "Baseline Appendix D",
        "paper §3.6 KM": "Baseline §3.6 KM",
        "paper Table 1 / §3.6 KM": "Baseline Table 1 / §3.6 KM",
    }
    out = replacements.get(s, s)
    return out.replace("&", r"\&") if out else "n.a."


def _fmt_cell(val: Any, *, decimals: int | None = None, missing: str = "n.r.") -> str:
    if val is None:
        return missing
    if isinstance(val, bool):
        return "Yes" if val else "No"
    if isinstance(val, int) and not isinstance(val, bool):
        return fmt_int(val)
    if isinstance(val, float):
        d = 4 if decimals is None else decimals
        # counts disguised as floats
        if abs(val - round(val)) < 1e-9 and abs(val) >= 10:
            return fmt_int(int(round(val)))
        return fmt_float(val, d)
    return str(val)


def _fmt_gap(gap: Any, *, decimals: int = 4) -> str:
    if gap is None:
        return "n.a."
    try:
        return fmt_signed_tex(float(gap), decimals)
    except (TypeError, ValueError):
        return str(gap)


def _hr_decimals(qid: str) -> int:
    if qid.startswith("hr_"):
        return 5
    if qid.startswith("n_") or qid in {"band_coverage", "n_rows"}:
        return 0
    return 4


# ── A01 protocol freeze (document, not a table) ───────────────────────────────

def build_a01(_: dict[str, Any] | None = None) -> Path:
    """A01_protocol_freeze.tex — continuous protocol document."""
    APPENDIX.mkdir(parents=True, exist_ok=True)
    out = APPENDIX / "A01_protocol_freeze.tex"
    body = PROTOCOL_MD.read_text(encoding="utf-8") if PROTOCOL_MD.exists() else ""

    def esc(s: str) -> str:
        return (
            s.replace("\\", r"\textbackslash{}")
            .replace("&", r"\&")
            .replace("%", r"\%")
            .replace("_", r"\_")
            .replace("#", r"\#")
        )

    lines = [
        "% Auto-generated by Block P — do not edit by hand.",
        "% Caption: Appendix A: Protocol Freeze Record (version 2026-07-12.c00.v5.1). "
        "Every methodological decision documented here was fixed before any test data were "
        "examined, except where explicitly noted (C00.4, \\S3.3). Rejected alternatives are "
        "listed for transparency.",
        "",
        r"\section*{Appendix A: Protocol Freeze ($2026$-$07$-$12$.c00.v5.1)}",
        r"\begin{description}",
    ]

    # Drop preamble / ## Hypotheses headers; keep ### blocks only.
    sections = re.split(r"\n(?=### )", body)
    for block in sections:
        block = block.strip()
        if not block.startswith("### "):
            continue
        first, _, rest = block.partition("\n")
        title = first[4:].strip()
        title_tex = esc(title).replace(r"\textbackslash{}", "")
        # undo over-escape of hyphens etc — title has no backslashes
        title_tex = title.replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")

        paras: list[str] = []
        table_rows: list[list[str]] = []
        in_table = False
        for ln in rest.splitlines():
            ln = ln.rstrip()
            if not ln or ln.startswith("## "):
                continue
            if ln.startswith("|"):
                in_table = True
                cells = [c.strip() for c in ln.strip("|").split("|")]
                if all(set(c) <= set("-: ") for c in cells):
                    continue  # separator
                table_rows.append([c.replace("_", r"\_") for c in cells])
                continue
            if in_table and table_rows:
                # flush ranking table as booktabs fragment
                from booktabs import to_booktabs

                paras.append(to_booktabs(table_rows[0], table_rows[1:], col_spec="ll"))
                table_rows = []
                in_table = False

            text = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", ln)
            text = text.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")
            if text.startswith("- "):
                paras.append(r"\item[--] " + text[2:])
            else:
                paras.append(text)
        if table_rows:
            from booktabs import to_booktabs

            paras.append(to_booktabs(table_rows[0], table_rows[1:], col_spec="ll"))

        lines.append(rf"\item[{title_tex}]")
        lines.extend(paras)
        lines.append("")

    lines.extend(
        [
            r"\item[C00.4 --- Reproduction admission criterion (declared a posteriori)]",
            r"A reproduced quantity within $0.01$ of its reported value is a \emph{strict} match; "
            r"within $(0.01,0.03]$ is \emph{approximate}; beyond $0.03$ is a \emph{reproduction gap}, "
            r"reported and retained. This criterion was fixed only after observing that all three "
            r"domains cleared it, and governs baseline \emph{admission}, not any hypothesis outcome "
            r"(see \S3.3).",
            r"{\footnotesize Rejected alternative: pre-registering a numerical tolerance before seeing "
            r"reproduction gaps --- rejected because none of the baselines pre-specify one; disclosing "
            r"the a-posteriori choice is required for honesty.}",
            "",
            r"\item[Rejected alternatives (transparency)]",
            r"\begin{itemize}",
            r"\item \textbf{C00.1:} Rank-permutation as the formal H1 gate --- rejected as unreachable "
            r"for $k\le 3$ models; retained as sensitivity-only. Primary test is subject-level "
            r"stratified bootstrap.",
            r"\item \textbf{C00.2:} (B) declare H1 N/A in Domain 1; (C) rank Harrell vs.\ Uno instead "
            r"of C-index vs.\ IPCW-IBS --- rejected in favour of the three-domain ranking objects "
            r"listed above.",
            r"\item \textbf{H5:} Uno AUC$(t)$ decay as primary --- rejected; pointwise IPCW Brier at "
            r"$\{12,24,36\}$ months is primary, with time-dependent AUC as sensitivity.",
            r"\end{itemize}",
            r"\end{description}",
            "",
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


# ── A02 per-domain reproduction detail ───────────────────────────────────────

def _build_a02_domain(domain: int, numbers: dict[str, Any]) -> tuple[Path, Path]:
    key = f"d0{domain}"
    doc = _load_repro(domain)
    cols = ["Quantity", "Reported", "Source", "Ours", "Gap"]
    rows: list[list[str]] = []
    for r in doc["rows"]:
        qid = str(r["quantity_id"])
        dec = _hr_decimals(qid)
        paper = r.get("paper_value")
        ours = r.get("ours_value")
        gap = r.get("gap")
        # string-valued cells (dates)
        if isinstance(paper, str) or isinstance(ours, str):
            rows.append(
                [
                    qid.replace("_", r"\_"),
                    "n.r." if paper is None else str(paper),
                    _clean_source(r.get("paper_source")),
                    "n.a." if ours is None else str(ours),
                    _fmt_gap(gap, decimals=dec if dec else 4),
                ]
            )
        else:
            rows.append(
                [
                    qid.replace("_", r"\_"),
                    _fmt_cell(paper, decimals=dec if dec else 4, missing="n.r."),
                    _clean_source(r.get("paper_source")),
                    _fmt_cell(ours, decimals=dec if dec else 4, missing="n.a."),
                    _fmt_gap(gap, decimals=4 if dec == 0 else dec),
                ]
            )

    stem = APPENDIX / f"A02_reproduction_detail_{key}"
    tex, csv = write_table(
        stem,
        cols,
        rows,
        caption=(
            f"Appendix B.{domain}: reproduction quantities for {DOMAIN_TITLES[key]}."
        ),
        table_note=FRASE_GUIA[key],
        col_spec="lllll",
    )

    # Append limitation table into the same .tex file
    lim_cols = ["\\#", "Limitation", "Paper says / omits", "What we did", "Implication"]
    lim_rows = LIMITATIONS[key]
    from booktabs import to_booktabs

    lim_body = to_booktabs(lim_cols, lim_rows, col_spec="lp{2.2cm}p{3.2cm}p{3.2cm}p{2.8cm}")
    extra = (
        f"\n% --- Limitations {key} ---\n"
        f"\\paragraph{{Limitations.}}\n"
        f"{lim_body}\n"
        f"\\smallskip\\noindent\\textit{{Frase-guia: {FRASE_GUIA[key]}}}\n"
    )
    tex.write_text(tex.read_text(encoding="utf-8") + extra, encoding="utf-8")

    # Also write a dedicated limitations CSV for audit
    lim_csv = APPENDIX / f"A02_limitations_{key}.csv"
    import csv as _csv

    with lim_csv.open("w", encoding="utf-8", newline="") as f:
        w = _csv.writer(f)
        w.writerow(["#", "Limitation", "Paper says / omits", "What we did", "Implication"])
        for row in lim_rows:
            w.writerow([re.sub(r"\\+", "", c).replace("\\,", ",") for c in row])

    return tex, csv


def build_a02(numbers: dict[str, Any] | None = None) -> list[tuple[Path, Path]]:
    n = numbers or load_numbers()
    APPENDIX.mkdir(parents=True, exist_ok=True)
    outs = [_build_a02_domain(d, n) for d in (1, 2, 3)]
    # Master tex that concatenates the three domain sections
    master = APPENDIX / "A02_reproduction_detail.tex"
    parts = [
        "% Auto-generated by Block P — do not edit by hand.",
        "% Caption: Appendix B: Per-domain reproduction detail (quantities, limitations, frase-guia).",
        "",
    ]
    for d, (tex, _) in enumerate(outs, start=1):
        parts.append(f"% === Domain {d} ===")
        parts.append(rf"\subsection*{{B.{d}\ {DOMAIN_TITLES[f'd0{d}']}}}")
        # strip the auto header comments from child
        body = tex.read_text(encoding="utf-8")
        body = re.sub(r"^%.*\n", "", body, count=2)
        parts.append(body)
        parts.append("")
    master.write_text("\n".join(parts), encoding="utf-8")
    return outs


# ── A03 anchor Figure 5 bias metrics ───────────────────────────────────────────

def _fmt_bias(x: float) -> str:
    ax = abs(float(x))
    if ax < 0.001:
        # scientific, 3 sig figs, TeX
        s = f"{float(x):.2e}"
        mant, exp = s.split("e")
        return rf"${mant}\times10^{{{int(exp)}}}$"
    return fmt_float(x, 4)


def build_a03(numbers: dict[str, Any] | None = None) -> tuple[Path, Path]:
    n = numbers or load_numbers()
    scenarios = [
        ("Random", "random"),
        ("Independent", "independent"),
        ("Dependent ($\\tau=0.25$)", "dep_tau25"),
        ("Dependent ($\\tau=0.50$)", "dep_tau50"),
        ("Dependent ($\\tau=0.75$)", "dep_tau75"),
    ]
    metrics = [
        ("bias\\_ci\\_harrell", "bias_ci_harrell"),
        ("bias\\_ci\\_uno", "bias_ci_uno"),
        ("bias\\_ipcw", "bias_ipcw"),
        ("bias\\_uncens", "bias_uncens"),
    ]
    cols = [
        "Censoring Scenario",
        "Metric",
        "Bias (ours)",
        "Bias (paper)",
        "Gap",
    ]
    rows: list[list[str]] = []
    gaps_abs: list[float] = []
    midrule_before: list[int] = []
    row_i = 0
    for s_i, (s_label, s_key) in enumerate(scenarios):
        if s_i > 0:
            midrule_before.append(row_i)
        for m_label, m_key in metrics:
            pref = f"anchor.figure5.{s_key}.{m_key}"
            ours = float(v(n, f"{pref}.ours"))
            paper = float(v(n, f"{pref}.paper"))
            gap = float(v(n, f"{pref}.gap"))
            gaps_abs.append(abs(gap))
            rows.append(
                [
                    s_label if m_key == "bias_ci_harrell" else "",
                    m_label,
                    _fmt_bias(ours),
                    _fmt_bias(paper),
                    _fmt_bias(gap),
                ]
            )
            row_i += 1

    # bold max |gap|
    max_g = max(gaps_abs) if gaps_abs else 0.0
    bold = []
    for g in gaps_abs:
        hit = abs(g - max_g) < 1e-15
        bold.append([False, False, False, False, hit])

    return write_table(
        APPENDIX / "A03_anchor_figure5",
        cols,
        rows,
        bold_mask=bold,
        midrule_before=midrule_before,
        caption=(
            "Appendix C: Full reproduction of the anchor paper's Figure 5 bias metrics "
            "(100 seeds). Each row reports the bias of a censored metric relative to its "
            "oracle counterpart, under the corresponding censoring scenario. See \\S3.4 and \\S4.2."
        ),
        table_note=(
            f"Maximum absolute gap highlighted in bold "
            f"(max $|\\mathrm{{gap}}|={max_g:.2e}$)."
        ),
    )


# ── A04 full reproduction D2 + D3 ─────────────────────────────────────────────

def build_a04_d02(numbers: dict[str, Any] | None = None) -> tuple[Path, Path]:
    _ = numbers or load_numbers()
    doc = _load_repro(2)
    cols = ["Quantity", "Reported", "Source", "Ours", "Gap"]
    rows = []
    bold = []
    headline = {"cindex_linear_test", "cindex_boosted_test"}
    for r in doc["rows"]:
        qid = str(r["quantity_id"])
        dec = _hr_decimals(qid)
        is_h = qid in headline
        paper, ours, gap = r.get("paper_value"), r.get("ours_value"), r.get("gap")
        if isinstance(paper, str) or isinstance(ours, str):
            row = [
                qid.replace("_", r"\_"),
                "n.r." if paper is None else str(paper),
                _clean_source(r.get("paper_source")),
                "n.a." if ours is None else str(ours),
                _fmt_gap(gap),
            ]
        else:
            row = [
                qid.replace("_", r"\_"),
                _fmt_cell(paper, decimals=dec if dec else 4, missing="n.r."),
                _clean_source(r.get("paper_source")),
                _fmt_cell(ours, decimals=dec if dec else 4, missing="n.a."),
                _fmt_gap(gap, decimals=4 if dec == 0 else max(dec, 4)),
            ]
        rows.append(row)
        bold.append([is_h] * 5)
    return write_table(
        APPENDIX / "A04_full_reproduction_d02",
        cols,
        rows,
        bold_mask=bold,
        caption=(
            "Appendix D.1: Complete Domain 2 reproduction quantities. "
            "Headline C-index rows (body Table 4) are bold."
        ),
    )


def build_a04_d03(numbers: dict[str, Any] | None = None) -> tuple[Path, Path]:
    n = numbers or load_numbers()
    cols = ["Community", "Feature set", "$\\theta$", "C (paper)", "C (ours)", "Gap"]
    order_comm = ["p", "ds", "cs"]
    order_feat = ["behavioural", "content", "combined"]
    order_theta = [24, 36]
    rows: list[list[str]] = []
    bold: list[list[bool]] = []
    midrule_before: list[int] = []
    row_i = 0
    max_abs = 0.0
    for ci, comm in enumerate(order_comm):
        if ci > 0:
            midrule_before.append(row_i)
        for feat in order_feat:
            for th in order_theta:
                key = f"repro.DOMAIN_03.cindex_{comm}_{feat}_theta{th}"
                paper = float(v(n, f"{key}.paper"))
                ours = float(v(n, f"{key}.ours"))
                gap = float(v(n, f"{key}.gap"))
                max_abs = max(max_abs, abs(gap))
                flag = abs(gap) > 0.005
                rows.append(
                    [
                        COMMUNITY[comm] if feat == "behavioural" and th == 24 else "",
                        FEATURE[feat] if th == 24 else "",
                        str(th),
                        fmt_float(paper, 4),
                        fmt_float(ours, 4),
                        fmt_signed_tex(gap, 4),
                    ]
                )
                bold.append([False, False, False, flag, flag, flag])
                row_i += 1

    # band_coverage final row
    band_ours = int(v(n, "repro.DOMAIN_03.band_coverage.ours"))
    band_paper = int(v(n, "repro.DOMAIN_03.band_coverage.paper"))
    band_gap = int(v(n, "repro.DOMAIN_03.band_coverage.gap"))
    midrule_before.append(row_i)
    rows.append(
        [
            "band\\_coverage",
            "---",
            "---",
            fmt_int(band_paper),
            fmt_int(band_ours),
            fmt_signed_tex(float(band_gap), 0),
        ]
    )
    bold.append([False] * 6)

    mean_gap = float(v(n, "repro.DOMAIN_03.cindex_mean_abs_gap"))
    note = (
        f"Mean absolute gap $= {mean_gap:.4f}$; "
        f"{band_ours} of 18 cells within the strict tier ($\\le 0.01$)."
    )
    if max_abs <= 0.005:
        note += " All 18 cells within $0.005$ of the published value."

    return write_table(
        APPENDIX / "A04_full_reproduction_d03",
        cols,
        rows,
        bold_mask=bold,
        midrule_before=midrule_before,
        caption=(
            "Appendix D: Complete reproduction of all 18 cells from the original paper's "
            "Table 8 (Domain 3, Stack Exchange). "
            f"Mean absolute gap $= {mean_gap:.4f}$; "
            f"{band_ours} of 18 cells fall within the strict reproduction tier ($\\le 0.01$). "
            "See \\S4.1."
        ),
        table_note=note,
    )


def build_a04(numbers: dict[str, Any] | None = None) -> list[tuple[Path, Path]]:
    n = numbers or load_numbers()
    return [build_a04_d02(n), build_a04_d03(n)]


BUILDERS: dict[str, Callable[..., Any]] = {
    "A01": build_a01,
    "A02": build_a02,
    "A03": build_a03,
    "A04": build_a04,
}


def build_all(numbers: dict[str, Any] | None = None) -> list[tuple[str, Any]]:
    n = numbers or load_numbers()
    APPENDIX.mkdir(parents=True, exist_ok=True)
    out: list[tuple[str, Any]] = []
    out.append(("A01", build_a01(n)))
    for i, (tex, csv) in enumerate(build_a02(n), start=1):
        out.append((f"A02-D{i}", (tex, csv)))
    out.append(("A03", build_a03(n)))
    for label, pair in zip(("A04-D2", "A04-D3"), build_a04(n)):
        out.append((label, pair))
    return out
