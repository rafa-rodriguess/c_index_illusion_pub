"""P02_paper_tables.py — Generate T01–T08 + appendix A01–A04
============================================================
Reads scalars from ``results/paper/numbers.json`` (P01) and
``results/reproduction/DOMAIN_0n_reproduction_table.json``.
Spec: ``artefatos.md``. Outputs:
  ``results/paper/tables/T0X_*.{tex,csv}``
  ``results/paper/appendix/A0X_*.{tex,csv}``

Execute (after P01):
    python -W default P02_paper_tables.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "results" / "paper" / "builders"))
sys.path.insert(0, str(ROOT / "results" / "paper" / "style"))

from appendix import build_all as build_appendix  # noqa: E402
from numbers_io import load_numbers  # noqa: E402
from tables import build_all as build_tables  # noqa: E402


def main() -> int:
    numbers = load_numbers()
    print(f"Loaded {len(numbers)} keys from numbers.json", flush=True)
    for name, tex, csv in build_tables(numbers):
        print(f"  {name}: {tex.relative_to(ROOT)} | {csv.name}", flush=True)
    for name, payload in build_appendix(numbers):
        if isinstance(payload, tuple):
            tex, csv = payload
            print(f"  {name}: {tex.relative_to(ROOT)} | {csv.name}", flush=True)
        else:
            print(f"  {name}: {Path(payload).relative_to(ROOT)}", flush=True)
    print("P02 done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
