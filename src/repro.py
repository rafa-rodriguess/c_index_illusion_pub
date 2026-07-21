"""
Reproducibility helpers — strict gates, D3 interpreter, machine fingerprint.
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

from src.metrics.io import utc_now, write_json

WAITING_EXIT_OK = 0
WAITING_EXIT_STRICT = 2


def strict_enabled(cli_strict: bool | None = None) -> bool:
    """True if --strict was passed or CDS_STRICT is 1/true/yes."""
    if cli_strict is True:
        return True
    if cli_strict is False:
        env = os.environ.get("CDS_STRICT", "").strip().lower()
        return env in {"1", "true", "yes", "on"}
    env = os.environ.get("CDS_STRICT", "").strip().lower()
    return env in {"1", "true", "yes", "on"}


def waiting_return(message: str, *, strict: bool | None = None) -> int:
    """
    Log-style WAITING exit code.

    Default (research / incremental runs): exit 0 so partial pipelines can proceed.
    Strict / paper mode (CDS_STRICT=1 or --strict): exit 2 so orchestrators fail-fast.
    """
    print(f"  WAITING: {message}", flush=True)
    if strict_enabled(strict):
        print("  STRICT: treating WAITING as failure (exit 2).", flush=True)
        return WAITING_EXIT_STRICT
    return WAITING_EXIT_OK


def add_strict_arg(parser) -> None:
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on WAITING / blocked gates (also: CDS_STRICT=1).",
    )


def project_root() -> Path:
    """Repository root (directory containing ``src/`` and ``requirements.lock``)."""
    return Path(__file__).resolve().parent.parent


def main_venv_python(root: Path | None = None) -> Path:
    """Path to ``.venv`` interpreter (may not exist yet)."""
    root = root or project_root()
    if os.name == "nt":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def resolve_main_python(root: Path | None = None) -> str:
    """
    Interpreter for the main trunk (everything except Domain-3 PySurvival RSF).

    Order: CDS_MAIN_PYTHON → ``<root>/.venv`` if present → ``sys.executable``.
    """
    root = root or project_root()
    env = os.environ.get("CDS_MAIN_PYTHON", "").strip()
    if env:
        p = Path(env)
        return str(p) if p.exists() else env
    venv_py = main_venv_python(root)
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def resolve_d3_python(default_fallback: str | None = None) -> str | None:
    """
    Interpreter for Domain-3 RSF train (patched pysurvival).

    Order: CDS_D3_PYTHON → results/logs/d3_python_path.txt → common conda paths
    → None (caller falls back to main trunk).
    """
    env = os.environ.get("CDS_D3_PYTHON", "").strip()
    if env:
        p = Path(env)
        return str(p) if p.exists() else env
    hint = project_root() / "results" / "logs" / "d3_python_path.txt"
    if hint.exists():
        line = hint.read_text().strip().splitlines()[:1]
        if line:
            p = Path(line[0].strip())
            if p.exists():
                return str(p)
    candidates = [
        default_fallback,
        "/opt/anaconda3/envs/d3-pysurvival/bin/python",
        str(Path.home() / "anaconda3/envs/d3-pysurvival/bin/python"),
        str(Path.home() / "miniconda3/envs/d3-pysurvival/bin/python"),
        str(Path.home() / "mambaforge/envs/d3-pysurvival/bin/python"),
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    return None


# Scripts that must use the Domain-3 PySurvival interpreter when available.
D3_PYSURVIVAL_SCRIPTS: frozenset[str] = frozenset(
    {
        "D02_DOMAIN_03_train_rsf.py",
    }
)


def resolve_script_python(script: str, *, root: Path | None = None) -> str:
    """Pick main-trunk or D3 interpreter for a pipeline script basename/path."""
    name = Path(script).name
    if name in D3_PYSURVIVAL_SCRIPTS:
        return resolve_d3_python() or resolve_main_python(root)
    return resolve_main_python(root)


def _sh(cmd: str) -> str | None:
    try:
        return subprocess.check_output(
            cmd, shell=True, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def _redact_hardware(text: str | None) -> str | None:
    if not text:
        return text
    text = re.sub(
        r"(Serial Number \(system\):)\s*\S+",
        r"\1 [REDACTED]",
        text,
    )
    text = re.sub(
        r"(Hardware UUID:)\s*\S+",
        r"\1 [REDACTED]",
        text,
    )
    return text


def collect_machine_fingerprint() -> dict[str, Any]:
    """Collect host hardware/OS/python inventory (PII redacted)."""
    mem_bytes = _sh("sysctl -n hw.memsize")
    mem_gb = None
    if mem_bytes and str(mem_bytes).isdigit():
        mem_gb = round(int(mem_bytes) / (1024**3), 2)

    pkgs: dict[str, str] = {}
    for import_name, key in [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scipy", "scipy"),
        ("sklearn", "scikit-learn"),
        ("sksurv", "scikit-survival"),
        ("lifelines", "lifelines"),
        ("xgboost", "xgboost"),
        ("torch", "torch"),
        ("joblib", "joblib"),
        ("pyarrow", "pyarrow"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
        ("statsmodels", "statsmodels"),
        ("survivaleval", "SurvivalEVAL"),
        ("SurvivalEVAL", "SurvivalEVAL"),
        ("pycop", "pycop"),
        ("optuna", "optuna"),
    ]:
        if key in pkgs:
            continue
        try:
            mod = __import__(import_name)
            pkgs[key] = str(getattr(mod, "__version__", "n/a"))
        except Exception as exc:  # noqa: BLE001
            if key not in pkgs:
                pkgs[key] = f"missing:{type(exc).__name__}"

    d3 = resolve_d3_python()
    d3_path = Path(d3) if d3 else None

    info: dict[str, Any] = {
        "generated_at_utc": utc_now(),
        "hostname": socket.gethostname(),
        "user": os.environ.get("USER") or os.environ.get("USERNAME"),
        "privacy": {
            "serial_and_uuid_redacted": True,
            "note": "Serial Number / Hardware UUID stripped for shareable logs.",
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "mac_ver": list(platform.mac_ver()),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "cpu": {
            "count_logical": os.cpu_count(),
            "brand": _sh("sysctl -n machdep.cpu.brand_string"),
            "physical_cores": _sh("sysctl -n hw.physicalcpu"),
            "logical_cores": _sh("sysctl -n hw.logicalcpu"),
        },
        "memory": {
            "memsize_bytes": mem_bytes,
            "memsize_gb": mem_gb,
        },
        "os": {
            "product_name": _sh("sw_vers -productName"),
            "product_version": _sh("sw_vers -productVersion"),
            "build_version": _sh("sw_vers -buildVersion"),
            "uname": _sh("uname -a"),
        },
        "disk_root": _sh("df -h / | tail -1"),
        "hardware": _redact_hardware(_sh("system_profiler SPHardwareDataType")),
        "displays": _sh("system_profiler SPDisplaysDataType"),
        "python_packages": pkgs,
        "d3_python": {
            "env_var_CDS_D3_PYTHON": os.environ.get("CDS_D3_PYTHON"),
            "resolved_path": d3,
            "exists": bool(d3_path and d3_path.exists()),
            "version": (
                _sh(f'"{d3}" -c "import sys; print(sys.version.split()[0])"')
                if d3_path and d3_path.exists()
                else None
            ),
        },
        "env_flags": {
            "CDS_STRICT": os.environ.get("CDS_STRICT"),
            "CDS_D3_PYTHON": os.environ.get("CDS_D3_PYTHON"),
        },
    }
    blob = "|".join(
        [
            info["hostname"] or "",
            info["platform"]["machine"] or "",
            str(info["cpu"].get("brand") or ""),
            str(info["memory"].get("memsize_bytes") or ""),
            info["os"].get("uname") or "",
        ]
    )
    info["machine_fingerprint_sha256"] = hashlib.sha256(blob.encode()).hexdigest()[:16]
    return info


def write_machine_fingerprint(path: Path) -> dict[str, Any]:
    payload = collect_machine_fingerprint()
    write_json(path, payload)
    return payload
