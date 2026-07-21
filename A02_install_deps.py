"""
A02_install_deps.py — Ensure ``.venv`` and install main-trunk dependencies
=========================================================================
Creates ``<repo>/.venv`` if missing, then ``pip install -r requirements.lock``
(or ``requirements.txt`` / ``CDS_REQUIREMENTS``) **into that venv**.

The Domain-3 PySurvival env is separate — use ``./bootstrap_envs.sh``
(or see ``domain3-abedi-2022/CODE_ACCESS.md``).

Execute:
    python A02_install_deps.py

Done when:
    - ``.venv`` exists and pip exits with code 0
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.repro import main_venv_python  # noqa: E402


def resolve_requirements() -> Path:
    override = os.environ.get("CDS_REQUIREMENTS", "").strip()
    if override:
        p = Path(override)
        return p if p.is_absolute() else ROOT / p
    lock = ROOT / "requirements.lock"
    if lock.exists():
        return lock
    return ROOT / "requirements.txt"


def ensure_main_venv() -> Path:
    """Create ``.venv`` with the invoking Python if it does not exist yet."""
    venv_py = main_venv_python(ROOT)
    if venv_py.exists():
        return venv_py
    print(f"Creating main-trunk venv: {ROOT / '.venv'}", flush=True)
    subprocess.run(
        [sys.executable, "-m", "venv", str(ROOT / ".venv")],
        check=True,
        cwd=str(ROOT),
    )
    if not venv_py.exists():
        raise RuntimeError(f"venv created but interpreter missing: {venv_py}")
    return venv_py


def install_deps() -> int:
    req = resolve_requirements()
    if not req.exists():
        print(f"ERROR: {req} not found.")
        return 1

    def log(msg: str = "") -> None:
        print(msg, flush=True)

    try:
        venv_py = ensure_main_venv()
    except Exception as exc:
        log(f"ERROR: could not create .venv: {exc}")
        return 1

    log(f"{'─' * 60}")
    log("INSTALLING DEPENDENCIES (main trunk → .venv)")
    log(f"{'─' * 60}")
    log(f"  Invoker : {sys.version.split()[0]}  ({sys.executable})")
    log(f"  Target  : {venv_py}")
    log(f"  File    : {req.name}" + (" (locked pins)" if req.name.endswith(".lock") else ""))
    log(f"{'─' * 60}")
    log()

    for cmd in (
        [str(venv_py), "-m", "pip", "install", "-U", "pip"],
        [str(venv_py), "-m", "pip", "install", "-r", str(req)],
    ):
        log("  $ " + " ".join(cmd))
        log()
        result = subprocess.run(cmd, cwd=str(ROOT))
        if result.returncode != 0:
            log(f"\nA02 failed — pip exited with code {result.returncode}.")
            return result.returncode

    log()
    log(f"{'─' * 60}")
    log("A02 complete — main trunk deps in .venv")
    if os.name == "nt":
        log(r"  Activate:  .venv\Scripts\activate")
    else:
        log("  Activate:  source .venv/bin/activate")
    log("  Jupyter:   select kernel = .venv/bin/python")
    log("  Domain-3:  ./bootstrap_envs.sh --d3   (separate conda env)")
    return 0


if __name__ == "__main__":
    raise SystemExit(install_deps())
