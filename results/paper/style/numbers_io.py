"""Load ``results/paper/numbers.json`` with typed helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
NUMBERS_PATH = ROOT / "results" / "paper" / "numbers.json"


def load_numbers(path: Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else NUMBERS_PATH
    doc = json.loads(p.read_text(encoding="utf-8"))
    return doc["numbers"] if "numbers" in doc else doc


def v(numbers: dict[str, Any], key: str) -> Any:
    if key not in numbers:
        raise KeyError(f"Missing numbers.json key: {key}")
    entry = numbers[key]
    if isinstance(entry, dict) and "value" in entry:
        return entry["value"]
    return entry


def fmt_float(x: float, decimals: int) -> str:
    return f"{float(x):.{decimals}f}"


def fmt_signed(x: float, decimals: int) -> str:
    val = float(x)
    sign = "+" if val >= 0 else "−"  # unicode minus for display; TeX uses $-$
    body = f"{abs(val):.{decimals}f}"
    return f"{sign}{body}"


def fmt_signed_tex(x: float, decimals: int) -> str:
    val = float(x)
    body = f"{abs(val):.{decimals}f}"
    if val >= 0:
        return f"+{body}"
    return f"$-${body}"


def fmt_p(p: float, *, ge_threshold: float = 0.001) -> str:
    """Scientific (3 sig figs) if p < 0.001; else 4 decimals. Exact 1.0 → 1.000."""
    p = float(p)
    if abs(p - 1.0) < 1e-12:
        return "1.000"
    # Prefer 4-decimal form when it rounds to ≥ 0.001 (e.g. 0.000999 → 0.0010)
    if round(p, 4) >= ge_threshold:
        return f"{p:.4f}"
    return f"{p:.2e}".replace("e-0", "e-").replace("e+0", "e+")


def fmt_p_tex(p: float, *, ge_threshold: float = 0.001) -> str:
    p = float(p)
    if abs(p - 1.0) < 1e-12:
        return "1.000"
    if round(p, 4) >= ge_threshold:
        return f"{p:.4f}"
    s = f"{p:.2e}"
    mant, exp = s.split("e")
    exp_i = int(exp)
    return rf"${mant}\times10^{{{exp_i}}}$"


def fmt_int(n: int) -> str:
    return f"{int(n):,}"


def yes_no(flag: bool) -> str:
    return "Yes" if flag else "No"
