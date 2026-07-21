"""
A04_machine_fingerprint.py — Register execution host (shareable)
===============================================================
Writes ``results/logs/machine_fingerprint.json`` with OS / CPU / RAM /
Python package versions. Serial Number and Hardware UUID are redacted.

Execute:
    python -W default A04_machine_fingerprint.py

Also invoked automatically from A03 when CDS_RECORD_MACHINE=1 (default on).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import cfg
from src.repro import write_machine_fingerprint


def main() -> int:
    out = cfg.DIRS["logs"] / "machine_fingerprint.json"
    payload = write_machine_fingerprint(out)
    print("─" * 60)
    print("A04 — MACHINE FINGERPRINT")
    print("─" * 60)
    print(f"  Wrote: {out.relative_to(cfg.ROOT)}")
    print(f"  fingerprint: {payload.get('machine_fingerprint_sha256')}")
    print(f"  host: {payload.get('hostname')}")
    cpu = payload.get("cpu") or {}
    mem = payload.get("memory") or {}
    print(f"  cpu: {cpu.get('brand')} ({cpu.get('logical_cores')} logical)")
    print(f"  memory_gb: {mem.get('memsize_gb')}")
    plat = payload.get("platform") or {}
    print(f"  python: {plat.get('python_version')} @ {plat.get('executable')}")
    print("  note: serial/UUID redacted for shareable logs")
    print("A04 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
