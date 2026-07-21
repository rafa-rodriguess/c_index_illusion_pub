"""
D01_DOMAIN_03_cv_protocol.py — Freeze 5-fold × 30-run protocol
=============================================================
Matches author RSF notebook CV quirks:

  1. Drop a 1% holdout via ``train_test_split(test_size=0.01, random_state=seed)``
  2. ``KFold(n_splits=5)`` **without** shuffle on the remaining 99%
  3. C-index is evaluated on the fold evaluation set (1% holdout unused for Table 8)

Seed rule (reproducible stand-in for notebook ``seed=None``):

  holdout_seed = RANDOM_SEED + run_id

Execute:
    python -W default D01_DOMAIN_03_cv_protocol.py
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import cfg

warnings.filterwarnings("default")


def log(msg: str = "") -> None:
    print(msg, flush=True)


def main() -> int:
    log("─" * 60)
    log("D01_DOMAIN_03 — CV PROTOCOL FREEZE")
    log("─" * 60)

    cv = cfg.DOMAIN_03["cv"]
    n_folds = int(cv["n_folds"])
    n_runs = int(cv["n_runs"])
    holdout_frac = float(cv.get("holdout_frac", 0.01))
    kfold_shuffle = bool(cv.get("kfold_shuffle", False))
    out_dir = cfg.DIRS["processed_d3"]
    out_dir.mkdir(parents=True, exist_ok=True)

    policy = {
        "stage": "D01_DOMAIN_03",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_folds": n_folds,
        "n_runs": n_runs,
        "holdout_frac": holdout_frac,
        "kfold_shuffle": kfold_shuffle,
        "random_seed": cfg.RANDOM_SEED,
        "fold_seed_rule": (
            "train_test_split(test_size=holdout_frac, random_state=RANDOM_SEED+run_id); "
            f"KFold(n_splits={n_folds}, shuffle={kfold_shuffle}) on remaining rows"
        ),
        "communities": {},
        "decisions": [
            {
                "id": "D01.1",
                "choice": f"{n_folds}-fold × {n_runs} runs after {holdout_frac:.0%} holdout",
                "source": "paper §5.2 + author RSF notebook",
            },
            {
                "id": "D01.2",
                "choice": "KFold without shuffle (author notebook default)",
                "source": "survival analysis using RSF.ipynb",
            },
            {
                "id": "D01.3",
                "choice": "holdout_seed = RANDOM_SEED + run_id (notebook used seed=None)",
                "source": "reproducibility stand-in",
            },
        ],
    }

    for code in cfg.DOMAIN_03["communities"]:
        path = out_dir / f"{code}_theta24.parquet"
        if not path.exists():
            log(f"ERROR: missing {path} — run D00 first")
            return 1
        n = len(pd.read_parquet(path, columns=["UserId"]))
        policy["communities"][code] = {"n": n}
        log(f"  {code}: n={n:,}")

    out = out_dir / "cv_policy.json"
    out.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(f"  Policy: {out.relative_to(cfg.ROOT)}")
    log("D01_DOMAIN_03 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
