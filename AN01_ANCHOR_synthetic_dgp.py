"""
AN01_ANCHOR_synthetic_dgp.py — Materialise anchor synthetic protocol
====================================================================
Writes the scenario manifest for the Lillelund et al. Weibull + Clayton
experiment (paper §5.1). Optionally smoke-imports the author ``dgp.py``
from the AN00 mirror when deps (torch, pycop, …) are available.

Does not run the 100-seed experiment (that is AN02).

Execute:
    python -W default AN01_ANCHOR_synthetic_dgp.py
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import cfg

warnings.filterwarnings("default")


def log(msg: str = "") -> None:
    print(msg, flush=True)


def _try_import_author_dgp(raw_dir: Path) -> dict:
    """Best-effort import of author DGP; never fails the stage."""
    if not (raw_dir / "dgp.py").exists():
        return {"status": "missing_code", "detail": "run AN00 first"}
    sys.path.insert(0, str(raw_dir))
    try:
        import dgp  # type: ignore  # noqa: F401

        return {
            "status": "import_ok",
            "module": "dgp",
            "classes": ["DGP_Weibull_linear", "SingleEventSyntheticDataLoader"],
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "import_blocked",
            "detail": str(exc),
            "hint": (
                "Install project requirements (includes torch, pycop, survivaleval) "
                "via python A02_install_deps.py; author pins are relaxed in "
                "requirements.txt for Domain-lane compatibility."
            ),
        }


def main() -> int:
    log("─" * 60)
    log("AN01 — ANCHOR SYNTHETIC DGP PROTOCOL")
    log("─" * 60)

    raw_dir: Path = cfg.ANCHOR["raw_dir"]
    out_dir: Path = cfg.ANCHOR["processed_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    if not (raw_dir / "dgp.py").exists():
        from src.repro import waiting_return

        return waiting_return(f"{raw_dir.relative_to(cfg.ROOT)}/dgp.py (run AN00).")

    import_info = _try_import_author_dgp(raw_dir)
    log(f"  Author dgp import: {import_info['status']}")
    if import_info.get("detail"):
        log(f"    detail: {import_info['detail']}")

    protocol = {
        "stage": "AN01",
        "role": "anchor_harness_check",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper": {
            "title": cfg.ANCHOR["title"],
            "arxiv": cfg.ANCHOR["arxiv"],
            "section": "§5.1 / Appendix D / Table 2 / Figure 5",
        },
        "model_fixed": cfg.ANCHOR["model"],
        "n_seeds": cfg.ANCHOR["n_seeds"],
        "copula_family": cfg.ANCHOR["copula"],
        "scenarios": cfg.ANCHOR["scenarios"],
        "table2_oracle_targets": cfg.ANCHOR["table2_oracle"],
        "author_code_dir": str(raw_dir.relative_to(cfg.ROOT)),
        "author_dgp_import": import_info,
        "metrics_to_match": [
            "oracle_cindex",
            "harrell_cindex",
            "uno_cindex",
            "oracle_ibs",
            "naive_ibs",
            "ipcw_ibs",
            "metric_error_vs_oracle",
        ],
        "note": (
            "Oracle metrics require true event times from the DGP — available "
            "only in this synthetic harness, not in Domain 01–03."
        ),
    }

    path = out_dir / "anchor_dgp_protocol.json"
    path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(f"  Wrote {path.relative_to(cfg.ROOT)}")
    log("AN01 complete — protocol frozen for harness run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
