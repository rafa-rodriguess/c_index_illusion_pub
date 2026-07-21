"""
D01_DOMAIN_01_split_ahmed.py — Domain 1 split policy (Ahmed & Green)
====================================================================
Materializes the **paper's** train/val/test policy for DOMAIN_01.

From Ahmed & Green §6 (DL path):
  - 80% train / 20% test
  - train further split 90/10 → train / validation
  - MaxAbs scaling fit on train only

Cox path (§4.1 / §7.1): GOF c-index 0.958 with lifelines.CoxPHFitter.
Paper does **not** state a hold-out for Cox. Smoke 2026-07-12 locked
population H6a (healthy calendar>7y ∪ all failed) as the fit cohort.

We therefore:
  - freeze DL split indices (for later probes / DL work)
  - mark ``cox_eval_mode = in_sample_gof`` on the H6a population

Prereq: ``data/processed/domain1/drives.*`` from D00.

Execute:
    python -W default D01_DOMAIN_01_split_ahmed.py
"""

from __future__ import annotations

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import cfg
from src.domain1_cox_cohort import ensure_cohort_columns

warnings.filterwarnings("default")


def log(msg: str = "") -> None:
    print(msg, flush=True)


def load_drives() -> pd.DataFrame:
    d = cfg.DIRS["processed_d1"]
    for name in ("drives.parquet", "drives.csv.gz", "drives.csv"):
        p = d / name
        if p.exists():
            if name.endswith(".parquet"):
                return ensure_cohort_columns(pd.read_parquet(p))
            return ensure_cohort_columns(pd.read_csv(p))
    raise FileNotFoundError(
        f"No drives table in {d}. Run D00_DOMAIN_01_features_backblaze.py first."
    )


def main() -> int:
    log("─" * 60)
    log("D01_DOMAIN_01 — SPLIT POLICY (Ahmed & Green)")
    log("─" * 60)

    df = load_drives()
    log(f"  Loaded {len(df):,} drives")

    rng = np.random.default_rng(cfg.RANDOM_SEED)
    n = len(df)
    perm = rng.permutation(n)
    test_frac = float(cfg.DOMAIN_01["dl_split"]["test_frac"])
    val_frac = float(cfg.DOMAIN_01["dl_split"]["val_frac_of_train"])
    n_test = int(round(n * test_frac))
    test_idx = np.sort(perm[:n_test])
    train_pool = perm[n_test:]
    n_val = int(round(len(train_pool) * val_frac))
    val_idx = np.sort(train_pool[:n_val])
    train_idx = np.sort(train_pool[n_val:])

    smart9_mask = df["cox_cohort_age_gt7"].astype(bool).to_numpy()
    h6a_mask = df["cox_fit_h6a"].astype(bool).to_numpy()
    serials = df["serial_number"].astype(str).to_numpy()

    payload = {
        "stage": "D01_DOMAIN_01",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": cfg.RANDOM_SEED,
        "n_drives": int(n),
        "dl_split": {
            **cfg.DOMAIN_01["dl_split"],
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_test": int(len(test_idx)),
            "train_serials": serials[train_idx].tolist(),
            "val_serials": serials[val_idx].tolist(),
            "test_serials": serials[test_idx].tolist(),
        },
        "cox": {
            "cohort_min_age_years": cfg.DOMAIN_01["cox_cohort_min_age_years"],
            "cox_cohort_age_basis": cfg.DOMAIN_01.get("cox_cohort_age_basis"),
            "n_h6a": int(h6a_mask.sum()),
            "n_h6a_failed": int(((df["event"] == 1) & h6a_mask).sum()),
            "n_h6a_healthy": int(((df["event"] == 0) & h6a_mask).sum()),
            "h6a_serials": serials[h6a_mask].tolist(),
            "n_flagged_age_gt7_smart9": int(smart9_mask.sum()),
            "n_age_gt7_failed_smart9": int(((df["event"] == 1) & smart9_mask).sum()),
            "n_age_gt7_healthy_smart9": int(((df["event"] == 0) & smart9_mask).sum()),
            "cox_fit_population": cfg.DOMAIN_01["cox_fit_population"],
            "eval_mode": "in_sample_gof",
            "eval_mode_rationale": (
                "Paper §7.1 reports Cox c-index 0.958 as goodness-of-fit; "
                "§6 80/20 split is for DeepNet/ML. Fit population H6a "
                "(calendar>7y healthy ∪ all failed) recovers C≈0.958 (smoke 2026-07-12). "
                "Author GitLab URL 404."
            ),
        },
        "decisions": [
            {
                "id": "D01.1",
                "choice": "DL split 80/20 then 90/10 of train → val (seed=RANDOM_SEED)",
                "source": "paper §6",
            },
            {
                "id": "D01.2",
                "choice": (
                    f"Cox reproduction uses {cfg.DOMAIN_01['cox_fit_population']} "
                    "(in-sample GOF); SMART9 both-class mask retained for audits only"
                ),
                "source": "paper §4.1 / §7.1 + smoke H6a",
            },
        ],
    }

    out_dir = cfg.DIRS["processed_d1"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "split_indices.json"
    np.savez_compressed(
        out_dir / "split_indices.npz",
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        cox_mask=h6a_mask,  # primary Cox mask = H6a
        cox_mask_smart9_both=smart9_mask,
    )
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    policy = {k: v for k, v in payload.items() if k != "dl_split"}
    policy["dl_split"] = {
        **cfg.DOMAIN_01["dl_split"],
        "n_train": payload["dl_split"]["n_train"],
        "n_val": payload["dl_split"]["n_val"],
        "n_test": payload["dl_split"]["n_test"],
    }
    policy["cox_summary"] = {
        k: payload["cox"][k]
        for k in (
            "cohort_min_age_years",
            "cox_cohort_age_basis",
            "n_h6a",
            "n_h6a_failed",
            "n_h6a_healthy",
            "n_flagged_age_gt7_smart9",
            "cox_fit_population",
            "eval_mode",
            "eval_mode_rationale",
        )
    }
    (out_dir / "split_policy.json").write_text(
        json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    log(f"  DL   train/val/test = {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    log(
        f"  H6a  n={int(h6a_mask.sum()):,} "
        f"(fail={payload['cox']['n_h6a_failed']:,}, "
        f"ok={payload['cox']['n_h6a_healthy']:,})"
    )
    log(f"  Cox fit population: {payload['cox']['cox_fit_population']}")
    log(f"  Wrote {out_json.relative_to(cfg.ROOT)}")
    log(f"  Wrote {(out_dir / 'split_indices.npz').relative_to(cfg.ROOT)}")
    log("D01_DOMAIN_01 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
