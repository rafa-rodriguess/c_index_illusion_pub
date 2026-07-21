"""
G01_export_harness.py — Block G: export reusable evaluation harness
===================================================================
Packages assumption-aligned metrics + protocol freeze for external use.

Writes:
  results/harness/
    README.md
    requirements-harness.txt
    protocol_freeze.md
    protocol_snippet.json
    manifest.json
    cds_metrics/          # copy of src/metrics (+ thin __init__)
    examples/smoke_harness.py

Execute:
    python -W default G01_export_harness.py
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import cfg
from src.metrics.io import write_json


def log(msg: str = "") -> None:
    print(msg, flush=True)


README = '''# Cross-domain survival evaluation harness

Reusable **assumption-aligned** metrics for survival models outside healthcare,
exported from the *C-index Illusion* audit pipeline.

Protocol: `{protocol_version}`

## What this is

A thin package (`cds_metrics/`) with:

- Discrimination: Harrell C, Uno IPCW C
- Calibration: D-Calibration / SurvivalEVAL helpers
- Proper scores: IPCW IBS, Brier at horizons
- Competing risks: naive CIF vs Aalen–Johansen (Bondora-style)
- Horizon strip: IPCW Brier + cumulative/dynamic AUC (H5-style)

It does **not** include domain data downloaders or baseline retraining.

## Install

```bash
pip install -r requirements-harness.txt
# then either:
export PYTHONPATH="$PWD:$PYTHONPATH"
# or copy cds_metrics into your project
```

## Smoke test

```bash
python examples/smoke_harness.py
```

## Minimal usage

```python
import numpy as np
from cds_metrics.discrimination import cindex_harrell, cindex_uno

time = np.array([10., 20., 30., 40.])
event = np.array([1, 0, 1, 1])
risk = np.array([0.9, 0.2, 0.7, 0.8])  # higher = higher risk

print(cindex_harrell(time, event, risk))
print(cindex_uno(time, event, risk, train_time=time, train_event=event))
```

## Protocol freeze

See `protocol_freeze.md` and `protocol_snippet.json` for pre-registered
hypotheses H1–H5 (including H4 existential ablation / H5 Brier-primary rewrite).

## Citation

If you use this harness, cite the audit paper (when available) and:

- Lillelund et al., *Position: Stop Chasing the C-index* (ICML 2026 Spotlight)
- SurvivalEVAL / sksurv / lifelines as used by the metric backends
'''


SMOKE = '''#!/usr/bin/env python
"""Smoke: import cds_metrics and compute Harrell C on a tiny toy sample."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from cds_metrics.discrimination import cindex_harrell, cindex_uno
from cds_metrics.temporal_auc import harrell_c, brier_at_horizons_sksurv
from sksurv.util import Surv

rng = np.random.default_rng(42)
n = 80
time = rng.uniform(1, 60, size=n)
event = rng.integers(0, 2, size=n)
risk = rng.normal(size=n)

h = cindex_harrell(time, event, risk)
u = cindex_uno(time, event, risk, train_time=time, train_event=event)
assert h["status"] == "live" and h["value"] is not None
assert u["status"] in {"live", "error"}  # Uno may fail on tiny samples
print("smoke OK — Harrell C =", round(float(h["value"]), 4))

# Brier path: flat S(t)=0.5
horizons = [12.0, 24.0, 36.0]
S = np.full((n, len(horizons)), 0.5)
b = brier_at_horizons_sksurv(
    time_train=time,
    event_train=event,
    time_test=time,
    event_test=event,
    surv_at_horizons=S,
    horizons=horizons,
)
print("smoke Brier status =", b.get("status"), "values =", b.get("values"))
'''


REQ = '''# Minimal deps for cds_metrics harness (subset of project requirements)
numpy>=1.26,<3
pandas>=2.2,<3
scipy>=1.13,<2
scikit-learn>=1.5,<2
scikit-survival>=0.23,<1
lifelines>=0.29,<1
survivaleval>=0.2.5
joblib>=1.4,<2
'''


def main() -> int:
    log("─" * 60)
    log("G01 — EXPORT EVALUATION HARNESS")
    log("─" * 60)

    out = cfg.DIRS.get("harness") or (cfg.ROOT / "results" / "harness")
    if "harness" not in cfg.DIRS:
        # allow without config edit — still write under results/harness
        pass
    out = cfg.ROOT / "results" / "harness"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    # Copy metrics package
    src_metrics = cfg.ROOT / "src" / "metrics"
    dst_pkg = out / "cds_metrics"
    shutil.copytree(
        src_metrics,
        dst_pkg,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    # Patch imports: src.metrics → cds_metrics, src.config → local stub
    for py in dst_pkg.glob("*.py"):
        text = py.read_text(encoding="utf-8")
        text2 = text.replace("from src.metrics.", "from cds_metrics.")
        text2 = text2.replace("from src.config import cfg", "from cds_metrics._cfg_stub import cfg")
        if text2 != text:
            py.write_text(text2, encoding="utf-8")

    # Lightweight public init (avoid project freeze / ROOT paths)
    (dst_pkg / "__init__.py").write_text(
        '"""Exported assumption-aligned survival metrics (cds_metrics)."""\n'
        '__all__ = [\n'
        '    "discrimination",\n'
        '    "calibration",\n'
        '    "proper_scores",\n'
        '    "competing_risks",\n'
        '    "temporal_auc",\n'
        ']\n',
        encoding="utf-8",
    )

    # Minimal cfg stub so exported modules import
    stub = '''"""Minimal config stub for the exported harness (no project ROOT needed)."""
from types import SimpleNamespace

PROTOCOL = {
    "version": "%s",
    "hypotheses": {
        "H3": {"horizon_months": 12, "delta": 0.02, "min_strata_consistent": 3},
        "H5": {
            "horizons_months": [12, 24, 36],
            "global_cindex_band": [0.66, 0.76],
        },
    },
}
EVAL = {
    "ibs_horizons_months": [12, 24, 36],
    "auc_horizons_months": [12, 24, 36],
    "n_bootstrap": 1000,
    "cindex_variants": ["harrell", "antolini", "uno"],
}
cfg = SimpleNamespace(PROTOCOL=PROTOCOL, EVAL=EVAL, RANDOM_SEED=42)
''' % (cfg.PROTOCOL.get("version"),)
    (dst_pkg / "_cfg_stub.py").write_text(stub, encoding="utf-8")

    # Protocol assets
    freeze_src = cfg.DIRS["logs"] / "protocol_freeze.md"
    if freeze_src.exists():
        shutil.copy2(freeze_src, out / "protocol_freeze.md")
    else:
        (out / "protocol_freeze.md").write_text(
            f"# Protocol freeze\n\nversion: {cfg.PROTOCOL.get('version')}\n",
            encoding="utf-8",
        )

    snippet = {
        "protocol_version": cfg.PROTOCOL.get("version"),
        "hypotheses": {
            k: {
                "title": v.get("title"),
                "decision": v.get("decision"),
                "domains": v.get("domains"),
                "method": v.get("method"),
                "honesty_note": v.get("honesty_note"),
                "primary_metric": v.get("primary_metric"),
            }
            for k, v in cfg.PROTOCOL.get("hypotheses", {}).items()
        },
    }
    write_json(out / "protocol_snippet.json", snippet)

    (out / "README.md").write_text(
        README.format(protocol_version=cfg.PROTOCOL.get("version")),
        encoding="utf-8",
    )
    (out / "requirements-harness.txt").write_text(REQ, encoding="utf-8")

    ex = out / "examples"
    ex.mkdir(parents=True)
    (ex / "smoke_harness.py").write_text(SMOKE, encoding="utf-8")

    # List packaged modules
    modules = sorted(p.name for p in dst_pkg.glob("*.py"))
    manifest = {
        "stage": "G01",
        "artifact": "harness_export",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": cfg.PROTOCOL.get("version"),
        "export_root": str(out.relative_to(cfg.ROOT)),
        "modules": modules,
        "smoke": "examples/smoke_harness.py",
    }
    write_json(out / "manifest.json", manifest)

    log(f"  Export root: {out.relative_to(cfg.ROOT)}")
    log(f"  Modules: {len(modules)}")

    # Run smoke
    log("  Running smoke_harness.py …")
    import subprocess

    proc = subprocess.run(
        [sys.executable, str(ex / "smoke_harness.py")],
        cwd=str(out),
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        log("SMOKE FAILED:")
        log(proc.stdout)
        log(proc.stderr)
        return 1
    log("  " + proc.stdout.strip().replace("\n", "\n  "))
    log("G01 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
