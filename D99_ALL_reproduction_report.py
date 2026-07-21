"""
D99_ALL_reproduction_report.py — Merge Fase A reproduction tables
=================================================================
Concatenates DOMAIN_01..03 paper assets into the unified gate artifact:

    results/reproduction/report.{json,csv,md}

Gate D→E (PIPELINE.md): for each domain, published value, ours, Δ, and
protocol deviations. Does **not** retrain or recompute metrics — only merges
existing ``DOMAIN_0n_reproduction_table.json`` files.

Prereq: all three lane gap scripts have written their tables.

Execute (when ready to close Fase A):
    python -W default D99_ALL_reproduction_report.py
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import cfg
from src.reproduction_table import SCHEMA_VERSION

DOMAIN_IDS = ("DOMAIN_01", "DOMAIN_02", "DOMAIN_03")
REPORT_SCHEMA = "2026-07-11.reproduction_report.v1"


def log(msg: str = "") -> None:
    print(msg, flush=True)


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


def _load_domain_table(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _flatten_rows(doc: dict[str, Any], *, main_only: bool = False) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in doc.get("rows") or []:
        if main_only and not r.get("in_main_table", True):
            continue
        out.append(
            {
                "domain_id": doc["domain_id"],
                "baseline": doc.get("baseline"),
                "doi": doc.get("doi"),
                "quantity_id": r.get("quantity_id"),
                "quantity_label": r.get("quantity_label"),
                "paper_value": r.get("paper_value"),
                "ours_value": r.get("ours_value"),
                "gap": r.get("gap"),
                "paper_source": r.get("paper_source"),
                "notes": r.get("notes", ""),
                "in_main_table": bool(r.get("in_main_table", True)),
            }
        )
    return out


def _domain_summary(doc: dict[str, Any]) -> dict[str, Any]:
    main = [r for r in doc.get("rows") or [] if r.get("in_main_table", True)]
    return {
        "domain_id": doc["domain_id"],
        "baseline": doc.get("baseline"),
        "doi": doc.get("doi"),
        "generated_at_utc": doc.get("generated_at_utc"),
        "schema_version": doc.get("schema_version"),
        "n_main_rows": len(main),
        "n_aux_rows": len(doc.get("rows") or []) - len(main),
        "n_protocol_deviations": len(doc.get("protocol_deviations") or []),
        "protocol_deviations": list(doc.get("protocol_deviations") or []),
        "meta": doc.get("meta") or {},
        "main_rows": [
            {
                "quantity_id": r.get("quantity_id"),
                "quantity_label": r.get("quantity_label"),
                "paper_value": r.get("paper_value"),
                "ours_value": r.get("ours_value"),
                "gap": r.get("gap"),
                "paper_source": r.get("paper_source"),
            }
            for r in main
        ],
    }


def _to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Cross-domain reproduction report (Fase A / D99)",
        "",
        f"- Generated (UTC): `{report['generated_at_utc']}`",
        f"- Report schema: `{report['schema_version']}`",
        f"- Source tables schema: `{report['source_schema_version']}`",
        f"- Gap definition: `{report['gap_definition']}`",
        f"- Gate D→E: `{report['gate_status']}`",
        "",
        "## Unified main table",
        "",
        "| Domain | Quantity | Paper | Ours | Gap (ours−paper) | Source |",
        "|--------|----------|------:|-----:|-----------------:|--------|",
    ]
    for r in report["rows"]:
        if not r.get("in_main_table", True):
            continue
        lines.append(
            f"| `{r['domain_id']}` | {r['quantity_label']} | {_fmt(r['paper_value'])} | "
            f"{_fmt(r['ours_value'])} | {_fmt(r['gap'])} | {r['paper_source']} |"
        )

    lines += ["", "## Per-domain protocol deviations", ""]
    for d in report["domains"]:
        lines.append(f"### `{d['domain_id']}` — {d['baseline']}")
        lines.append("")
        if d["protocol_deviations"]:
            for item in d["protocol_deviations"]:
                lines.append(f"- {item}")
        else:
            lines.append("- *(none listed)*")
        lines.append("")

    missing = report.get("missing_domains") or []
    if missing:
        lines += [
            "## Missing inputs",
            "",
            "D99 cannot close the D→E gate until these tables exist:",
            "",
        ]
        for mid in missing:
            lines.append(f"- `{mid}_reproduction_table.json`")
        lines.append("")

    lines += [
        "## Paper usage",
        "",
        report.get("paper_usage", ""),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    log("─" * 60)
    log("D99_ALL — UNIFIED REPRODUCTION REPORT")
    log("─" * 60)

    out_dir = cfg.ROOT / "results" / "reproduction"
    out_dir.mkdir(parents=True, exist_ok=True)

    docs: list[dict[str, Any]] = []
    missing: list[str] = []
    for domain_id in DOMAIN_IDS:
        path = out_dir / f"{domain_id}_reproduction_table.json"
        doc = _load_domain_table(path)
        if doc is None:
            missing.append(domain_id)
            log(f"  WAITING: {path.relative_to(cfg.ROOT)}")
            continue
        docs.append(doc)
        log(f"  Loaded: {path.relative_to(cfg.ROOT)} ({len(doc.get('rows') or [])} rows)")

    gate_closed = len(missing) == 0
    all_rows: list[dict[str, Any]] = []
    for doc in docs:
        all_rows.extend(_flatten_rows(doc, main_only=False))

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "artifact": "reproduction_report",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_schema_version": SCHEMA_VERSION,
        "gap_definition": "gap = ours_value - paper_value",
        "gate_status": "ready_for_E" if gate_closed else "blocked_missing_domains",
        "domains_expected": list(DOMAIN_IDS),
        "domains_present": [d["domain_id"] for d in docs],
        "missing_domains": missing,
        "domains": [_domain_summary(d) for d in docs],
        "rows": all_rows,
        "paper_usage": (
            "Unified Fase A asset for Results / Reproduction Protocol. "
            "Prefer domain-level DOMAIN_0n_reproduction_table for appendix detail; "
            "use this report for the cross-domain summary. "
            "Do not re-type numbers into the paper — cite these artifacts."
        ),
    }

    json_path = out_dir / "report.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    csv_cols = [
        "domain_id",
        "baseline",
        "doi",
        "quantity_id",
        "quantity_label",
        "paper_value",
        "ours_value",
        "gap",
        "paper_source",
        "notes",
        "in_main_table",
    ]
    csv_path = out_dir / "report.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_cols)
        writer.writeheader()
        for r in all_rows:
            writer.writerow({k: r.get(k) for k in csv_cols})

    md_path = out_dir / "report.md"
    md_path.write_text(_to_markdown(report), encoding="utf-8")

    for label, path in (("json", json_path), ("csv", csv_path), ("md", md_path)):
        log(f"  Wrote {label:4s}: {path.relative_to(cfg.ROOT)}")

    if not gate_closed:
        from src.repro import waiting_return

        return waiting_return(
            f"D99 incomplete — missing {missing}. "
            "Re-run after all DOMAIN_0n gap scripts finish."
        )

    log("D99 complete — gate D→E ready (models remain those under results/models/).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
