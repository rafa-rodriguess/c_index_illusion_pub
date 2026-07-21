"""
A03_env_check.py — Verify environment and package versions
==========================================================
Imports required packages, prints versions, and reports missing ones.
Run after A02_install_deps.py.

Execute:
    python A03_env_check.py

Done when:
    - All packages in REQUIRED_PYTHON import without error
    - Versions are printed for the audit trail
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

REQUIRED_PYTHON = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("sksurv", "scikit-survival"),
    ("lifelines", "lifelines"),
    ("xgboost", "xgboost"),
    ("requests", "requests"),
    ("pyarrow", "pyarrow"),
    ("openpyxl", "openpyxl"),
    ("joblib", "joblib"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    # Block AN — anchor harness (Lillelund position-cindex)
    ("torch", "torch"),
    ("SurvivalEVAL", "SurvivalEVAL"),  # PyPI / import name (capitalization)
    ("pycop", "pycop"),
    ("statsmodels", "statsmodels"),
]


def check_python_packages() -> list[str]:
    missing: list[str] = []
    print(f"\n{'─' * 60}")
    print("PYTHON PACKAGES")
    print(f"{'─' * 60}")
    print(f"  {'Import':<16} {'Package':<18} {'Version':<12} Status")
    print(f"  {'─' * 16} {'─' * 18} {'─' * 12} {'─' * 6}")

    for import_name, display_name in REQUIRED_PYTHON:
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "n/a")
            print(f"  {import_name:<16} {display_name:<18} {version:<12} ✓")
        except ImportError as exc:
            print(f"  {import_name:<16} {display_name:<18} {'N/A':<12} ✗  ({exc})")
            missing.append(display_name)

    return missing


def main() -> int:
    print(f"Python {sys.version.split()[0]}  ({sys.executable})")
    missing = check_python_packages()

    print(f"\n{'─' * 60}")
    if missing:
        print("MISSING:", ", ".join(missing))
        print("Run:  python A02_install_deps.py")
        print("A03 incomplete.")
        return 1

    print("All required packages are available.")

    # Record host fingerprint (serial/UUID redacted) unless explicitly disabled.
    import os

    if os.environ.get("CDS_RECORD_MACHINE", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }:
        try:
            from src.config import cfg
            from src.repro import write_machine_fingerprint

            out = cfg.DIRS["logs"] / "machine_fingerprint.json"
            payload = write_machine_fingerprint(out)
            print(
                f"Machine fingerprint: {payload.get('machine_fingerprint_sha256')} "
                f"→ {out.relative_to(cfg.ROOT)}"
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Machine fingerprint skipped: {exc}")

    print("A03 complete — environment OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
