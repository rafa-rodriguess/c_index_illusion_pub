"""
A02b_bootstrap_d3_env.py — Ensure Domain-3 conda env ``d3-pysurvival``
=====================================================================
Creates (or reuses) the PySurvival 0.1.2 interpreter used **only** by
``D02_DOMAIN_03_train_rsf.py``. Delegates to ``./bootstrap_envs.sh --d3-only``.

Requirements:
  - ``conda`` on PATH (Miniconda / Anaconda / Mambaforge)
  - On some hosts, a one-line C++ ``tp_print`` patch — see
    ``domain3-abedi-2022/CODE_ACCESS.md``

Execute (from notebook or CLI):
    python -W default A02b_bootstrap_d3_env.py

Exit codes:
  0 — env ready (or already present with importable pysurvival)
  2 — WAITING treated as failure under CDS_STRICT=1 / --strict
  0 — WAITING logged (non-strict) when conda/build unavailable
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.repro import (  # noqa: E402
    add_strict_arg,
    resolve_d3_python,
    waiting_return,
)


def _write_hint(d3_py: str) -> None:
    hint = ROOT / "results" / "logs" / "d3_python_path.txt"
    hint.parent.mkdir(parents=True, exist_ok=True)
    hint.write_text(d3_py + "\n")
    os.environ["CDS_D3_PYTHON"] = d3_py
    print(f"  CDS_D3_PYTHON={d3_py}", flush=True)
    print(f"  wrote {hint.relative_to(ROOT)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_strict_arg(parser)
    args = parser.parse_args()

    existing = resolve_d3_python()
    if existing:
        probe = subprocess.run(
            [existing, "-c", "import pysurvival"],
            capture_output=True,
            text=True,
        )
        if probe.returncode == 0:
            print(f"A02b: d3-pysurvival already ready → {existing}", flush=True)
            _write_hint(existing)
            return 0
        print(
            f"A02b: found {existing} but pysurvival import failed; re-bootstrapping…",
            flush=True,
        )

    script = ROOT / "bootstrap_envs.sh"
    if not script.exists():
        return waiting_return(
            f"missing {script.name} — cannot bootstrap Domain-3 env",
            strict=args.strict,
        )

    if not os.access(script, os.X_OK):
        script.chmod(script.stat().st_mode | 0o111)

    print("A02b: running ./bootstrap_envs.sh --d3-only …", flush=True)
    proc = subprocess.run(
        ["bash", str(script), "--d3-only"],
        cwd=str(ROOT),
    )
    if proc.returncode != 0:
        return waiting_return(
            "Domain-3 bootstrap failed (conda missing or pysurvival build/patch). "
            "See domain3-abedi-2022/CODE_ACCESS.md",
            strict=args.strict,
        )

    d3 = resolve_d3_python()
    if not d3:
        return waiting_return(
            "bootstrap finished but resolve_d3_python() still empty",
            strict=args.strict,
        )

    _write_hint(d3)
    print("A02b complete — Domain-3 interpreter ready.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
