"""
P00_collect_artifacts.py — Paper collection (traceable inventory)
===============================================================
Sequentially inventories every artefact the manuscript may cite.
Writes a provenance manifest with paths, SHA-256, mtimes, and status.

**Block P rule:** P01 (and future P02/P03) read **only** from this collection
(and the files it points to). No paper number is hard-coded in P scripts.

Writes:
  results/paper/collection/manifest.json
  results/paper/collection/index.md
  results/logs/P00_collect_artifacts.md

Execute:
    python -W default P00_collect_artifacts.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import PROTOCOL, cfg
from src.metrics.io import utc_now, write_json

# Ordered sources — sequential, traceable. Paths relative to repo root.
PAPER_SOURCES: list[dict[str, Any]] = [
    {"id": "protocol_freeze", "rel": "results/logs/protocol_freeze.json", "required": True, "block": "C00"},
    {"id": "frozen_manifest", "rel": "results/ladder/frozen_models_manifest.json", "required": True, "block": "E00"},
    {"id": "ladder_summary", "rel": "results/ladder/summary.json", "required": True, "block": "E06"},
    {"id": "d01_discrimination", "rel": "results/ladder/d01_discrimination.json", "required": True, "block": "E01"},
    {"id": "d02_discrimination", "rel": "results/ladder/d02_discrimination.json", "required": True, "block": "E01"},
    {"id": "d03_discrimination", "rel": "results/ladder/d03_discrimination.json", "required": True, "block": "E01"},
    {"id": "d01_calibration", "rel": "results/ladder/d01_calibration.json", "required": True, "block": "E02"},
    {"id": "d02_calibration", "rel": "results/ladder/d02_calibration.json", "required": True, "block": "E02"},
    {"id": "d03_calibration", "rel": "results/ladder/d03_calibration.json", "required": True, "block": "E02"},
    {"id": "d01_scores", "rel": "results/ladder/d01_scores.json", "required": True, "block": "E03"},
    {"id": "d02_scores", "rel": "results/ladder/d02_scores.json", "required": True, "block": "E03"},
    {"id": "d03_scores", "rel": "results/ladder/d03_scores.json", "required": True, "block": "E03"},
    {"id": "F00_h4_ablation", "rel": "results/probes/F00_h4_ablation.json", "required": False, "block": "F"},
    {"id": "F00_sens1_loo", "rel": "results/probes/F00_sens1_leave_one_out.json", "required": True, "block": "F"},
    {"id": "F01_h3", "rel": "results/probes/F01_h3_competing_risks.json", "required": True, "block": "F"},
    {"id": "F02_h5", "rel": "results/probes/F02_h5_temporal.json", "required": True, "block": "F"},
    {"id": "F03_report", "rel": "results/probes/report.json", "required": True, "block": "F"},
    {"id": "H01", "rel": "results/probes/H01_ranking_inversion.json", "required": True, "block": "G"},
    {"id": "H04", "rel": "results/probes/H04_paired_bootstrap_loo.json", "required": True, "block": "G"},
    {"id": "G03", "rel": "results/reproduction/G03_holm_family.json", "required": True, "block": "G"},
    {"id": "G02_verdict", "rel": "results/reproduction/FINAL_verdict_table.json", "required": True, "block": "G"},
    {"id": "DOMAIN_01_repro", "rel": "results/reproduction/DOMAIN_01_reproduction_table.json", "required": True, "block": "D"},
    {"id": "DOMAIN_02_repro", "rel": "results/reproduction/DOMAIN_02_reproduction_table.json", "required": True, "block": "D"},
    {"id": "DOMAIN_03_repro", "rel": "results/reproduction/DOMAIN_03_reproduction_table.json", "required": False, "block": "D"},
    {"id": "ANCHOR_table2", "rel": "results/reproduction/ANCHOR_table2_compare.json", "required": False, "block": "AN"},
    {"id": "ANCHOR_figure5", "rel": "results/reproduction/ANCHOR_figure5_compare.json", "required": False, "block": "AN"},
    {"id": "leakage_audit", "rel": "results/logs/leakage_controls_audit.json", "required": False, "block": "G00"},
    {"id": "harness_manifest", "rel": "results/harness/manifest.json", "required": False, "block": "G01"},
]


def log(msg: str = "") -> None:
    print(msg, flush=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    log("─" * 60)
    log("P00 — COLLECT PAPER ARTEFACTS (sequential inventory)")
    log("─" * 60)

    root = cfg.ROOT
    out_dir = cfg.DIRS["paper"] / "collection"
    out_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    n_ok = n_missing = n_optional_missing = 0

    for i, src in enumerate(PAPER_SOURCES, start=1):
        rel = src["rel"]
        path = root / rel
        log(f"  [{i:02d}/{len(PAPER_SOURCES)}] {src['id']} ← {rel}")
        entry: dict[str, Any] = {
            "id": src["id"],
            "block": src["block"],
            "rel": rel,
            "required": bool(src["required"]),
            "order": i,
        }
        if not path.exists():
            entry["status"] = "missing"
            if src["required"]:
                n_missing += 1
                log("       MISSING (required)")
            else:
                n_optional_missing += 1
                log("       missing (optional)")
        else:
            digest = _sha256(path)
            entry.update(
                {
                    "status": "ok",
                    "bytes": path.stat().st_size,
                    "mtime_utc": utc_now(),  # collection time; file mtime below
                    "mtime_epoch": path.stat().st_mtime,
                    "sha256": digest,
                }
            )
            n_ok += 1
            log(f"       ok  sha256={digest[:12]}…  bytes={entry['bytes']}")
        entries.append(entry)

    payload = {
        "stage": "P00",
        "role": "paper_collection_manifest",
        "generated_at_utc": utc_now(),
        "protocol_version": PROTOCOL["version"],
        "n_sources": len(PAPER_SOURCES),
        "n_ok": n_ok,
        "n_missing_required": n_missing,
        "n_missing_optional": n_optional_missing,
        "complete_required": n_missing == 0,
        "sources": entries,
        "rule": (
            "P01 (and future table/figure steps) must read numbers only from these "
            "artefacts (via this manifest). No hard-coded paper scalars."
        ),
    }

    man_path = out_dir / "manifest.json"
    write_json(man_path, payload)

    lines = [
        "# P00 — Paper artefact collection",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        f"Protocol: `{payload['protocol_version']}`",
        f"Required complete: **{payload['complete_required']}** "
        f"(ok={n_ok}, missing_required={n_missing}, missing_optional={n_optional_missing})",
        "",
        "| # | id | block | status | sha256 | path |",
        "|---|----|-------|--------|--------|------|",
    ]
    for e in entries:
        sha = (e.get("sha256") or "—")[:12]
        if e.get("sha256"):
            sha += "…"
        lines.append(
            f"| {e['order']} | `{e['id']}` | {e['block']} | {e['status']} | `{sha}` | `{e['rel']}` |"
        )
    lines += [
        "",
        "## Rule",
        "",
        payload["rule"],
        "",
    ]
    md = "\n".join(lines) + "\n"
    (out_dir / "index.md").write_text(md, encoding="utf-8")
    cfg.DIRS["logs"].mkdir(parents=True, exist_ok=True)
    (cfg.DIRS["logs"] / "P00_collect_artifacts.md").write_text(md, encoding="utf-8")

    log(f"Wrote {man_path}")
    log(
        f"P00 complete — required_complete={payload['complete_required']} "
        f"ok={n_ok}/{len(PAPER_SOURCES)}"
    )
    return 0 if n_missing == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
