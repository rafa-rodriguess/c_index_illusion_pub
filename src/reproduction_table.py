"""
reproduction_table.py — Paper-ready mixed reproduction-gap artifacts
====================================================================
Canonical schema for every DOMAIN_0n lane:

    results/reproduction/DOMAIN_0{n}_reproduction_table.{json,csv,md,tex}

Each row pairs a quantity extracted from the baseline paper with the value
from our pipeline and the gap (ours − paper). These tables are first-class
paper assets (roadmap §7 Fase A / §13 Results).
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "2026-07-11.reproduction_table.v1"

COLUMNS = [
    "quantity_id",
    "quantity_label",
    "paper_value",
    "ours_value",
    "gap",
    "paper_source",
    "notes",
    "in_main_table",
]


def gap_value(ours: float | int | None, paper: float | int | None) -> float | None:
    if ours is None or paper is None:
        return None
    return float(ours) - float(paper)


def row(
    quantity_id: str,
    quantity_label: str,
    paper_value: Any,
    ours_value: Any,
    paper_source: str,
    notes: str = "",
    in_main_table: bool = True,
) -> dict[str, Any]:
    g = None
    try:
        if paper_value is not None and ours_value is not None:
            g = float(ours_value) - float(paper_value)
    except (TypeError, ValueError):
        g = None
    return {
        "quantity_id": quantity_id,
        "quantity_label": quantity_label,
        "paper_value": paper_value,
        "ours_value": ours_value,
        "gap": g,
        "paper_source": paper_source,
        "notes": notes,
        "in_main_table": bool(in_main_table),
    }


def build_document(
    *,
    domain_id: str,
    baseline: str,
    doi: str,
    rows: list[dict[str, Any]],
    protocol_deviations: list[str] | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "reproduction_table",
        "domain_id": domain_id,
        "baseline": baseline,
        "doi": doi,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gap_definition": "gap = ours_value - paper_value",
        "rows": rows,
        "protocol_deviations": protocol_deviations or [],
        "paper_usage": (
            "Primary paper asset for Fase A: paste main-table rows into "
            "Results / Reproduction Protocol. Deviations go in appendix or "
            "limitations. D99 merges DOMAIN_01..03 into the unified table."
        ),
    }
    if extra_meta:
        doc["meta"] = extra_meta
    return doc


def _fmt(v: Any, digits: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def write_reproduction_table(doc: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    """Write JSON / CSV / Markdown / LaTeX. Returns path map."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{doc['domain_id']}_reproduction_table"
    paths: dict[str, Path] = {}

    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["json"] = json_path

    csv_path = out_dir / f"{stem}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for r in doc["rows"]:
            writer.writerow({k: r.get(k) for k in COLUMNS})
    paths["csv"] = csv_path

    md_path = out_dir / f"{stem}.md"
    md_path.write_text(_to_markdown(doc), encoding="utf-8")
    paths["md"] = md_path

    tex_path = out_dir / f"{stem}.tex"
    tex_path.write_text(_to_latex(doc), encoding="utf-8")
    paths["tex"] = tex_path

    return paths


def _to_markdown(doc: dict[str, Any]) -> str:
    lines = [
        f"# Reproduction table — `{doc['domain_id']}`",
        "",
        f"- Baseline: **{doc['baseline']}**",
        f"- DOI: `{doc['doi']}`",
        f"- Generated (UTC): `{doc['generated_at_utc']}`",
        f"- Gap definition: `{doc['gap_definition']}`",
        f"- Schema: `{doc['schema_version']}`",
        "",
        "## Main table (paper-ready)",
        "",
        "| Quantity | Paper | Ours | Gap (ours−paper) | Source |",
        "|----------|------:|-----:|-----------------:|--------|",
    ]
    for r in doc["rows"]:
        if not r.get("in_main_table", True):
            continue
        lines.append(
            f"| {r['quantity_label']} | {_fmt(r['paper_value'])} | "
            f"{_fmt(r['ours_value'])} | {_fmt(r['gap'])} | {r['paper_source']} |"
        )

    aux = [r for r in doc["rows"] if not r.get("in_main_table", True)]
    if aux:
        lines += [
            "",
            "## Auxiliary quantities",
            "",
            "| Quantity | Paper | Ours | Gap | Notes |",
            "|----------|------:|-----:|----:|-------|",
        ]
        for r in aux:
            lines.append(
                f"| {r['quantity_label']} | {_fmt(r['paper_value'])} | "
                f"{_fmt(r['ours_value'])} | {_fmt(r['gap'])} | {r.get('notes','')} |"
            )

    if doc.get("protocol_deviations"):
        lines += ["", "## Protocol deviations", ""]
        for d in doc["protocol_deviations"]:
            lines.append(f"- {d}")

    lines += ["", "## Paper usage", "", doc.get("paper_usage", ""), ""]
    return "\n".join(lines)


def _to_latex(doc: dict[str, Any]) -> str:
    """booktabs snippet for Results / Reproduction Protocol."""
    domain = doc["domain_id"].replace("_", r"\_")
    lines = [
        "% Auto-generated reproduction gap table — do not edit by hand",
        f"% domain: {doc['domain_id']}  schema: {doc['schema_version']}",
        r"\begin{table}[t]",
        r"  \centering",
        rf"  \caption{{Reproduction gap — {doc['baseline']} ({domain}). "
        r"Gap $=$ ours $-$ paper.}",
        rf"  \label{{tab:repro-{doc['domain_id'].lower()}}}",
        r"  \begin{tabular}{lrrr}",
        r"    \toprule",
        r"    Quantity & Paper & Ours & Gap \\",
        r"    \midrule",
    ]
    for r in doc["rows"]:
        if not r.get("in_main_table", True):
            continue
        label = str(r["quantity_label"]).replace("_", r"\_").replace("%", r"\%")
        lines.append(
            f"    {label} & {_fmt(r['paper_value'])} & {_fmt(r['ours_value'])} "
            f"& {_fmt(r['gap'])} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)
