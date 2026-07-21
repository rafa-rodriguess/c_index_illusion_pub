"""
P03_paper_figures.py — Generate F01–F05 (vector PDF)
====================================================
F01: Graphviz compile. F02–F05: matplotlib + SciencePlots.
Reads scalars from ``results/paper/numbers.json``. Spec: ``artefatos.md``.

Execute (after P01):
    python -W default P03_paper_figures.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Writable matplotlib cache (sandbox / CI)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / "paper" / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "results" / "paper" / "builders"))
sys.path.insert(0, str(ROOT / "results" / "paper" / "style"))

from figures import build_all  # noqa: E402
from numbers_io import load_numbers  # noqa: E402


def main() -> int:
    numbers = load_numbers()
    print(f"Loaded {len(numbers)} keys from numbers.json", flush=True)
    for name, path in build_all(numbers):
        print(f"  {name}: {path.relative_to(ROOT)}", flush=True)
    print("P03 done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
