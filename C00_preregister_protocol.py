"""
C00_preregister_protocol.py — Freeze pre-registered protocol to disk
====================================================================
Writes an immutable snapshot of ``cfg.PROTOCOL`` (roadmap §C00 + §8)
to ``results/logs/protocol_freeze.{json,md}`` **before** any test-set
evaluation.

If a freeze file already exists:
  - identical content hash → idempotent success (skipped)
  - different content → fail (bump ``PROTOCOL["version"]`` to intentionally refresh)

Execute:
    python C00_preregister_protocol.py

Done when:
    - freeze JSON/MD present
    - content SHA-256 recorded
    - exit code 0 under ``python -W default``
"""

from __future__ import annotations

import hashlib
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import cfg

warnings.filterwarnings("default")


def log(msg: str = "") -> None:
    print(msg, flush=True)


def canonical_payload() -> dict[str, Any]:
    """Build the freeze document (deterministic key order via sort_keys later)."""
    return {
        "stage": "C00",
        "protocol_version": cfg.PROTOCOL["version"],
        "frozen_at_utc": None,  # filled at write time for new freezes
        "random_seed": cfg.RANDOM_SEED,
        "globals": cfg.PROTOCOL["globals"],
        "eval": {
            "alpha": cfg.EVAL["alpha"],
            "n_bootstrap": cfg.EVAL["n_bootstrap"],
            "n_permutation": cfg.EVAL["n_permutation"],
            "cindex_variants": cfg.EVAL["cindex_variants"],
            "d_calibration_bins": cfg.EVAL["d_calibration_bins"],
            "auc_horizons_months": cfg.EVAL["auc_horizons_months"],
            "ibs_horizons_months": cfg.EVAL["ibs_horizons_months"],
            "n_cv_folds": cfg.EVAL["n_cv_folds"],
            "n_cv_repeats": cfg.EVAL["n_cv_repeats"],
        },
        "freeze_decisions": cfg.PROTOCOL["freeze_decisions"],
        "hypotheses": cfg.PROTOCOL["hypotheses"],
        "reproduction_targets": cfg.PROTOCOL["reproduction_targets"],
        "h4_ablation_smart_ids": list(cfg.H4_ABLATION_SMART),
        "domains": {
            k: {
                "name": v["name"],
                "baseline": v["baseline"],
                "doi": v["doi"],
                "reported_cindex": v["reported_cindex"],
            }
            for k, v in cfg.DOMAINS.items()
        },
        "anchor": {
            "venue": cfg.ANCHOR["venue"],
            "arxiv": cfg.ANCHOR["arxiv"],
            "doi": cfg.ANCHOR["doi"],
        },
        "roadmap_sections": cfg.PROTOCOL["roadmap_sections"],
    }


def content_bytes(payload: dict[str, Any]) -> bytes:
    # Exclude volatile timestamp from the integrity hash
    body = {k: v for k, v in payload.items() if k not in {"frozen_at_utc", "content_sha256"}}
    return json.dumps(body, sort_keys=True, indent=2).encode("utf-8") + b"\n"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_protocol() -> None:
    p = cfg.PROTOCOL
    assert p["version"], "PROTOCOL.version required"
    assert set(p["hypotheses"]) >= {"H1", "H2", "H3", "H4", "H5", "H_meta"}
    ranks = p["freeze_decisions"]["C00.2"]["rankings"]
    assert len(ranks["domain1"]) >= 2
    assert len(ranks["domain2"]) >= 2
    assert len(ranks["domain3"]) >= 2
    assert cfg.H4_ABLATION_SMART == [5, 197, 198]
    assert cfg.EVAL["auc_horizons_months"] == [12, 24, 36]
    assert p["hypotheses"]["H3"]["delta"] == 0.02
    assert p["hypotheses"]["H4"]["delta_c_threshold"] == 0.03
    assert p["freeze_decisions"]["C00.3"]["meta_family_size"] == 5
    assert p["hypotheses"]["H_meta"]["min_rejected_of_five"] == 3


def to_markdown(payload: dict[str, Any], digest: str) -> str:
    lines = [
        f"# Protocol freeze — `{payload['protocol_version']}`",
        "",
        f"- Frozen at (UTC): `{payload['frozen_at_utc']}`",
        f"- Content SHA-256: `{digest}`",
        f"- Random seed: `{payload['random_seed']}`",
        f"- α = {payload['globals']['alpha']}, "
        f"B = {payload['globals']['n_bootstrap']}, "
        f"permutations = {payload['globals']['n_permutation']}",
        "",
        "## C00 freeze decisions",
        "",
        "### C00.1 — H1 decision rule",
        payload["freeze_decisions"]["C00.1"]["official"],
        "",
        f"- Binary inversion cross-domain: "
        f"**{payload['freeze_decisions']['C00.1']['binary_inversion_cross_domain']}**",
        "",
        "### C00.2 — H1 ranking objects",
        "",
        "| Domain | Models |",
        "|--------|--------|",
    ]
    for dom, models in payload["freeze_decisions"]["C00.2"]["rankings"].items():
        lines.append(f"| {dom} | {', '.join(models)} |")
    lines.extend(
        [
            "",
            "### C00.3 — Multiple-testing family",
            f"- Inner: {payload['freeze_decisions']['C00.3']['inner']}",
            f"- Outer: {payload['freeze_decisions']['C00.3']['outer']}",
            f"- Meta family size: {payload['freeze_decisions']['C00.3']['meta_family_size']}",
            "",
            "## Hypotheses (decision rules)",
            "",
        ]
    )
    for key in ("H1", "H2", "H3", "H4", "H5", "H_meta"):
        h = payload["hypotheses"][key]
        lines.extend(
            [
                f"### {key} — {h['title']}",
                f"- **H0:** {h['H0']}",
                f"- **H1:** {h['H1']}",
            ]
        )
        if "statistic" in h:
            lines.append(f"- **Statistic:** {h['statistic']}")
        if "decision" in h:
            lines.append(f"- **Decision:** {h['decision']}")
        lines.append("")

    lines.extend(
        [
            "## Reproduction targets",
            "",
            "| Domain | Target |",
            "|--------|--------|",
        ]
    )
    for dom, t in payload["reproduction_targets"].items():
        lines.append(f"| {dom} | `{json.dumps(t, sort_keys=True)}` |")
    lines.extend(
        [
            "",
            f"## H1 D1 companion ablation SMART ids (cox_ablated only): "
            f"`{payload['h4_ablation_smart_ids']}`",
            "",
            "Not an H4 claim — H4 exhibits are LOO survey + H04 bootstrap.",
            "",
            "This freeze is authoritative for Blocks D–G. "
            "Do not alter without bumping `PROTOCOL['version']`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    log_dir = cfg.DIRS["logs"]
    log_dir.mkdir(parents=True, exist_ok=True)
    json_path = log_dir / "protocol_freeze.json"
    md_path = log_dir / "protocol_freeze.md"

    log(f"{'─' * 60}")
    log("C00 — PREREGISTER / PROTOCOL FREEZE")
    log(f"{'─' * 60}")
    log(f"  Version : {cfg.PROTOCOL['version']}")
    log(f"  Out     : {json_path.relative_to(cfg.ROOT)}")
    log(f"{'─' * 60}")

    validate_protocol()
    payload = canonical_payload()
    digest = sha256_bytes(content_bytes(payload))

    if json_path.exists():
        existing = json.loads(json_path.read_text(encoding="utf-8"))
        old_body = {k: v for k, v in existing.items() if k not in {"frozen_at_utc", "content_sha256"}}
        # Compare against current body without timestamp
        new_body = {k: v for k, v in payload.items() if k != "frozen_at_utc"}
        if json.dumps(old_body, sort_keys=True) == json.dumps(new_body, sort_keys=True):
            log("\n  Existing freeze matches current PROTOCOL — skipped (idempotent).")
            log(f"  SHA-256 : {existing.get('content_sha256', digest)}")
            log("C00 complete — protocol already frozen.")
            return 0
        old_ver = existing.get("protocol_version")
        new_ver = payload["protocol_version"]
        if old_ver == new_ver:
            log("\nERROR: protocol_freeze.json exists but differs from cfg.PROTOCOL.")
            log(f"  On disk version : {old_ver}")
            log(f"  Current version : {new_ver}")
            log("  Bump PROTOCOL['version'] in src/config.py to intentionally refresh.")
            return 1
        # Intentional amendment: archive prior freeze, then write the new one.
        archive = log_dir / f"protocol_freeze_{old_ver}.json"
        archive_md = log_dir / f"protocol_freeze_{old_ver}.md"
        archive.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")
        if md_path.exists():
            archive_md.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")
        log(f"\n  Amendment refresh: {old_ver} → {new_ver}")
        log(f"  Archived prior freeze → {archive.name}")

    payload["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["content_sha256"] = digest
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(to_markdown(payload, digest), encoding="utf-8")

    log(f"\n  Wrote JSON : {json_path.relative_to(cfg.ROOT)}")
    log(f"  Wrote MD   : {md_path.relative_to(cfg.ROOT)}")
    log(f"  SHA-256    : {digest}")
    log("C00 complete — protocol frozen (do not edit without version bump).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
