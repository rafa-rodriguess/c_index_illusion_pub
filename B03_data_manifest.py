"""
B03_data_manifest.py — Auditable inventory of downloaded raw data
=================================================================
Scans ``data/raw/`` for artefacts from B00–B02, computes SHA-256 and
file metadata, and writes a machine-readable manifest plus a short
Markdown summary under ``results/logs/``.

Does not download anything. Fails if any expected domain raw payload
is missing (Backblaze zips, Bondora loan xlsx, Stack Exchange features).

Execute:
    python B03_data_manifest.py

Done when:
    - ``results/logs/data_manifest.json`` written
    - ``results/logs/data_manifest.md`` written
    - Exit code 0 under ``python -W default``
"""

from __future__ import annotations

import hashlib
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import cfg

warnings.filterwarnings("default")

CHUNK = 1 << 20  # 1 MiB


def log(msg: str = "") -> None:
    print(msg, flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(CHUNK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def iter_raw_files(raw_root: Path) -> list[Path]:
    files: list[Path] = []
    for path in sorted(raw_root.rglob("*")):
        if not path.is_file():
            continue
        if path.name == ".gitkeep":
            continue
        files.append(path)
    return files


def source_hint(rel: str) -> str:
    if rel.startswith("backblaze/"):
        name = Path(rel).name
        return cfg.DATA_URLS["backblaze_zip_template"].format(filename=name)
    if rel.startswith("bondora/"):
        if rel.endswith(cfg.BONDORA["loan_local_name"]):
            return cfg.DATA_URLS["bondora_loan_xlsx"]
        return cfg.DATA_URLS["bondora_stats_page"]
    if rel.startswith("stackexchange/"):
        # Strip leading stackexchange/ to recover repo-relative path
        repo_rel = rel[len("stackexchange/") :]
        return cfg.STACKEXCHANGE["url_template"].format(path=repo_rel)
    return "unknown"


def validate_expected(raw_root: Path) -> list[str]:
    """Return list of human-readable problems (empty if OK)."""
    problems: list[str] = []

    bb = raw_root / "backblaze"
    for name in cfg.BACKBLAZE["zip_filenames"]:
        p = bb / name
        if not p.exists() or p.stat().st_size <= 0:
            problems.append(f"missing Backblaze zip: {name}")

    bondora_loan = raw_root / "bondora" / cfg.BONDORA["loan_local_name"]
    if not bondora_loan.exists() or bondora_loan.stat().st_size <= 0:
        problems.append(f"missing Bondora loan file: {cfg.BONDORA['loan_local_name']}")

    se = raw_root / "stackexchange"
    for comm in cfg.STACKEXCHANGE["communities"]:
        feat = se / "processed_data" / comm / "user_features.csv"
        if not feat.exists() or feat.stat().st_size <= 0:
            problems.append(f"missing Stack Exchange features: {feat.relative_to(raw_root)}")

    return problems


def main() -> int:
    raw_root = cfg.DIRS["raw"]
    log_dir = cfg.DIRS["logs"]
    log_dir.mkdir(parents=True, exist_ok=True)

    log(f"{'─' * 60}")
    log("B03 — DATA MANIFEST")
    log(f"{'─' * 60}")
    log(f"  Raw root : {raw_root.relative_to(cfg.ROOT)}")
    log(f"{'─' * 60}")

    problems = validate_expected(raw_root)
    if problems:
        log("\nERROR: expected raw artefacts missing:")
        for p in problems:
            log(f"  - {p}")
        return 1

    files = iter_raw_files(raw_root)
    log(f"\nHashing {len(files)} files…")

    entries = []
    total_bytes = 0
    for i, path in enumerate(files, 1):
        rel = str(path.relative_to(raw_root))
        size = path.stat().st_size
        total_bytes += size
        digest = sha256_file(path)
        mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
        entry = {
            "path": rel,
            "bytes": size,
            "sha256": digest,
            "mtime_utc": mtime,
            "source_url": source_hint(rel),
        }
        entries.append(entry)
        if i % 10 == 0 or i == len(files):
            log(f"  [{i}/{len(files)}] {rel}  ({size / 1e6:.1f} MB)")

    by_domain = {
        "backblaze": [e for e in entries if e["path"].startswith("backblaze/")],
        "bondora": [e for e in entries if e["path"].startswith("bondora/")],
        "stackexchange": [e for e in entries if e["path"].startswith("stackexchange/")],
    }

    manifest = {
        "stage": "B03",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "raw_root": str(raw_root.relative_to(cfg.ROOT)),
        "n_files": len(entries),
        "total_bytes": total_bytes,
        "domains": {
            name: {
                "n_files": len(items),
                "bytes": sum(e["bytes"] for e in items),
            }
            for name, items in by_domain.items()
        },
        "notes": {
            "bondora_repayments": (
                "Not present. See results/logs/b01_bondora_status.json. "
                "Not required for roadmap Cox/ladder/CR probe."
            ),
            "stackexchange_source": (
                "Author GitHub artefacts (processed + raw_data), not Archive.org dumps."
            ),
        },
        "files": entries,
    }

    json_path = log_dir / "data_manifest.json"
    md_path = log_dir / "data_manifest.md"
    json_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Data manifest (B03)",
        "",
        f"- Generated (UTC): `{manifest['timestamp_utc']}`",
        f"- Files: **{manifest['n_files']}**",
        f"- Total size: **{total_bytes / 1e9:.2f} GB**",
        "",
        "| Domain | Files | Bytes |",
        "|--------|------:|------:|",
    ]
    for name, meta in manifest["domains"].items():
        lines.append(f"| {name} | {meta['n_files']} | {meta['bytes']:,} |")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            f"- Bondora repayments: {manifest['notes']['bondora_repayments']}",
            f"- Stack Exchange: {manifest['notes']['stackexchange_source']}",
            "",
            "## Files",
            "",
            "| Path | MB | SHA-256 (12) |",
            "|------|---:|--------------|",
        ]
    )
    for e in entries:
        lines.append(
            f"| `{e['path']}` | {e['bytes'] / 1e6:.1f} | `{e['sha256'][:12]}` |"
        )
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    log(f"\n{'─' * 60}")
    log(f"  JSON : {json_path.relative_to(cfg.ROOT)}")
    log(f"  MD   : {md_path.relative_to(cfg.ROOT)}")
    log(
        f"  Totals: {len(entries)} files, {total_bytes / 1e9:.2f} GB  "
        f"(bb={manifest['domains']['backblaze']['n_files']}, "
        f"bondora={manifest['domains']['bondora']['n_files']}, "
        f"se={manifest['domains']['stackexchange']['n_files']})"
    )
    log("B03 complete — data manifest written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
