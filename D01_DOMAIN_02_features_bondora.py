"""
D01_DOMAIN_02_features_bondora.py — Domain 2 features / survival + §3.2 preprocess
================================================================================
Consumes aligned LoanData (D00), builds §3.2 survival labels, then applies
``src.domain2_preprocess`` (engineered features, rare-cat filter, one-hot).

Execute:
    python -W default D01_DOMAIN_02_features_bondora.py
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
from src.domain2_preprocess import preprocess_for_modeling

warnings.filterwarnings("default")


def log(msg: str = "") -> None:
    print(msg, flush=True)


def load_aligned() -> tuple[pd.DataFrame, Path]:
    aligned = cfg.DIRS["interim_d2"] / "LoanData_aligned.parquet"
    if aligned.exists():
        return pd.read_parquet(aligned), aligned
    raise FileNotFoundError(
        f"Missing {aligned.relative_to(cfg.ROOT)} — run "
        "D00_DOMAIN_02_align_loan_vintage.py first."
    )


def build_survival(df: pd.DataFrame, obs_end: pd.Timestamp) -> pd.DataFrame:
    out = df.copy()
    out["LoanDate"] = pd.to_datetime(out["LoanDate"], errors="coerce")
    out["DefaultDate"] = pd.to_datetime(out.get("DefaultDate"), errors="coerce")
    out["ContractEndDate"] = pd.to_datetime(out.get("ContractEndDate"), errors="coerce")
    out["MaturityDate_Original"] = pd.to_datetime(
        out.get("MaturityDate_Original"), errors="coerce"
    )

    term_days = int(round(cfg.DOMAIN_02["loan_duration_months"] * 365.25 / 12))
    defaulted = out["DefaultDate"].notna()
    early = (
        (~defaulted)
        & out["ContractEndDate"].notna()
        & out["MaturityDate_Original"].notna()
        & (out["ContractEndDate"] < out["MaturityDate_Original"])
    )

    t_default = (out["DefaultDate"] - out["LoanDate"]).dt.days
    t_term = pd.Series(term_days, index=out.index)
    t_obs = (obs_end - out["LoanDate"]).dt.days
    t_contract = (out["ContractEndDate"] - out["LoanDate"]).dt.days

    survival = t_obs.copy().astype(float)
    event = pd.Series(0, index=out.index, dtype=int)

    m = defaulted & t_default.notna()
    survival.loc[m] = np.minimum(t_default.loc[m].clip(lower=1), t_obs.loc[m].clip(lower=1))
    event.loc[m] = 1

    survival.loc[early] = float(term_days)
    event.loc[early] = 0

    other = ~(defaulted | early)
    cand = pd.concat(
        [
            t_term.loc[other],
            t_obs.loc[other],
            t_contract.loc[other].fillna(np.inf),
        ],
        axis=1,
    ).min(axis=1)
    survival.loc[other] = cand.clip(lower=1)
    event.loc[other] = 0

    out["duration_days"] = survival
    out["event"] = event
    out["early_repayment"] = early.astype(int)
    out["obs_end"] = obs_end
    return out


def main() -> int:
    log("─" * 60)
    log("D01_DOMAIN_02 — FEATURES + §3.2 PREPROCESS")
    log("─" * 60)

    df, path = load_aligned()
    log(f"  Source: {path.relative_to(cfg.ROOT)}  rows={len(df):,}  cols={len(df.columns)}")

    obs_end = pd.Timestamp(cfg.DOMAIN_02["paper_retrieve_date"])
    if "ReportAsOfEOD" in df.columns:
        reported = pd.to_datetime(df["ReportAsOfEOD"], errors="coerce").max()
        if pd.notna(reported):
            obs_end = reported
    log(f"  Observation end (as-of): {obs_end.date()}")

    df = build_survival(df, obs_end)
    # Keep LoanId for traceability through preprocess meta
    if "LoanId" not in df.columns and "LoanNumber" in df.columns:
        df["LoanId"] = df["LoanNumber"]

    out, prep_report = preprocess_for_modeling(df)
    log(
        f"  Preprocess: {prep_report['n_input']:,} → {prep_report['n_complete']:,} "
        f"(rare−{prep_report['n_dropped_rare']:,}, miss−{prep_report['n_dropped_missing']:,})"
    )
    log(f"  Design features: {prep_report['n_features_design']}")

    out_dir = cfg.DIRS["processed_d2"]
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet = out_dir / "loans.parquet"
    out.to_parquet(parquet, index=False)
    out.to_csv(out_dir / "loans.csv.gz", index=False, compression="gzip")

    feat_path = out_dir / "design_features.json"
    feat_path.write_text(
        json.dumps(
            {
                "features": prep_report["feature_names"],
                "numeric": prep_report["numeric_features"],
                "categorical": prep_report["categorical_features"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    decisions = {
        "stage": "D01_DOMAIN_02",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(path.relative_to(cfg.ROOT)),
        "data_source_url": cfg.DOMAIN_02["data_source_url"],
        "obs_end": str(obs_end.date()),
        "n_events": int(out["event"].sum()),
        "n_early_repayment": int(out["early_repayment"].sum()),
        "loan_date_min": str(pd.to_datetime(out["LoanDate"]).min().date()),
        "loan_date_max": str(pd.to_datetime(out["LoanDate"]).max().date()),
        "preprocess": prep_report,
        "can_form_paper_2020_test": bool(
            pd.to_datetime(out["LoanDate"]).max() >= pd.Timestamp("2020-01-01")
        ),
        "decisions": [
            {
                "id": "D01.1",
                "choice": "Survival labels per §3.2 + footnote 8 (early repay → full term)",
                "source": "paper §3.2",
            },
            {
                "id": "D01.2",
                "choice": "§3.2 preprocess: DTI modeled, RepaymentRatio, Country_Lang, one-hot",
                "source": "paper §3.2 / Table 4 auction-time features",
            },
            {
                "id": "D01.3",
                "choice": "Exclude Bondora Rating/EL/PD/LGD from Cox design (fair vs platform)",
                "source": "Table 4 Cox_2 spirit + rating comparison protocol",
            },
        ],
    }
    dec_path = cfg.DIRS["logs"] / "d01_domain_02_decisions.json"
    dec_path.write_text(json.dumps(decisions, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    log(f"  Wrote {parquet.relative_to(cfg.ROOT)}")
    log(f"  Events={decisions['n_events']:,} / {prep_report['n_complete']:,}")
    log("D01_DOMAIN_02 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
