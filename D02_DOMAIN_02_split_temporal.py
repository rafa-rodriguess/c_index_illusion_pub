"""
D02_DOMAIN_02_split_temporal.py — Temporal split (Bone-Winkel & Reichenbach)
===========================================================================
Paper §3.3: split date 2020-01-01; train/val = 90/10 of pre-split loans
(random); test = **all loans originated in calendar year 2020**.

Execute:
    python -W default D02_DOMAIN_02_split_temporal.py
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

warnings.filterwarnings("default")


def log(msg: str = "") -> None:
    print(msg, flush=True)


def main() -> int:
    log("─" * 60)
    log("D02_DOMAIN_02 — TEMPORAL SPLIT")
    log("─" * 60)

    path = cfg.DIRS["processed_d2"] / "loans.parquet"
    if not path.exists():
        log("ERROR: run D01 first.")
        return 1
    df = pd.read_parquet(path)
    df["LoanDate"] = pd.to_datetime(df["LoanDate"])

    paper_split = pd.Timestamp(cfg.DOMAIN_02["temporal_split_date_paper"])
    adapted = pd.Timestamp(cfg.DOMAIN_02["temporal_split_date_adapted"])
    test_end = pd.Timestamp(cfg.DOMAIN_02["test_year_end"])
    max_date = df["LoanDate"].max()

    if max_date >= paper_split:
        split = paper_split
        adapted_flag = False
    else:
        split = adapted
        adapted_flag = True
        log(
            f"  WARNING: data end {max_date.date()} < paper split {paper_split.date()} "
            f"→ using adapted split {split.date()}"
        )

    rng = np.random.default_rng(cfg.RANDOM_SEED)
    # Paper: test = originations in 2020 only (not all post-split years)
    if not adapted_flag:
        pre = df["LoanDate"] < split
        post = (df["LoanDate"] >= split) & (df["LoanDate"] <= test_end)
    else:
        pre = df["LoanDate"] < split
        post = df["LoanDate"] >= split

    pre_idx = np.flatnonzero(pre.to_numpy())
    post_idx = np.flatnonzero(post.to_numpy())
    rng.shuffle(pre_idx)
    n_val = int(round(len(pre_idx) * float(cfg.DOMAIN_02["train_val_pre_split"]["val_frac"])))
    val_idx = np.sort(pre_idx[:n_val])
    train_idx = np.sort(pre_idx[n_val:])
    test_idx = np.sort(post_idx)

    loan_dates = df["LoanDate"]
    duration = df["duration_days"].to_numpy().astype(float)
    event = df["event"].to_numpy().astype(int)
    days_to_split = (split - loan_dates).dt.days.to_numpy()

    for idx in np.concatenate([train_idx, val_idx]):
        cap = max(int(days_to_split[idx]), 1)
        if duration[idx] > cap:
            duration[idx] = cap
            event[idx] = 0

    df_out = df.copy()
    df_out["duration_days_split"] = duration
    df_out["event_split"] = event
    df_out["split_role"] = "none"
    df_out.loc[df_out.index[train_idx], "split_role"] = "train"
    df_out.loc[df_out.index[val_idx], "split_role"] = "val"
    df_out.loc[df_out.index[test_idx], "split_role"] = "test"
    df_out.to_parquet(cfg.DIRS["processed_d2"] / "loans_split.parquet", index=False)

    payload = {
        "stage": "D02_DOMAIN_02",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": cfg.RANDOM_SEED,
        "paper_split_date": str(paper_split.date()),
        "split_date_used": str(split.date()),
        "test_year_end": str(test_end.date()),
        "adapted_split": adapted_flag,
        "test_definition": (
            "calendar_year_2020" if not adapted_flag else "all_post_split"
        ),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "n_train_events": int(event[train_idx].sum()),
        "n_test_events": int(df.loc[df.index[test_idx], "event"].sum()) if len(test_idx) else 0,
        "decisions": [
            {
                "id": "D02.1",
                "choice": "Temporal split; train/val 90/10 random pre-split",
                "source": "paper §3.3",
            },
            {
                "id": "D02.2",
                "choice": "Test = 2020 originations only (not post-2020)",
                "source": "paper §3.3",
            },
            {
                "id": "D02.3",
                "choice": "Re-censor train/val durations at split date",
                "source": "paper §3.3",
            },
            {
                "id": "D02.4",
                "choice": f"split_date_used={split.date()} (adapted={adapted_flag})",
                "source": "data vintage",
                "flag": "adapted_split" if adapted_flag else None,
            },
        ],
    }
    out = cfg.DIRS["processed_d2"] / "split_policy.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    np.savez_compressed(
        cfg.DIRS["processed_d2"] / "split_indices.npz",
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )

    log(f"  Split date used: {split.date()}  (adapted={adapted_flag})")
    log(f"  Test definition: {payload['test_definition']}")
    log(f"  train/val/test = {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    log(f"  Wrote {out.relative_to(cfg.ROOT)}")
    if len(test_idx) == 0:
        log("  ERROR: empty test set — cannot evaluate ratings.")
        return 1
    log("D02_DOMAIN_02 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
