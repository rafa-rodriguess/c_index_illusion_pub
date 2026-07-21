"""
D00_DOMAIN_02_align_loan_vintage.py — Align Bondora LoanData to paper vintage
==============================================================================
Lane: DOMAIN_02 (Bone-Winkel & Reichenbach 2024).

Takes the public Kaggle LoanData dump and filters **rows + columns** to
approximate the authors' extract retrieved **2024-01-03**:

  - LoanDuration == 36 (3-year loans, §3.1)
  - LoanDate in [2014-01-01, 2020-12-31]  (train window + 2020 test year only)
  - Outcomes after paper retrieve date are treated as unobserved
  - Drop post-auction / leakage-heavy columns (§3.2 step 1 spirit)
  - Keep auction-time + survival keys needed by D01–D03

Source (logged):
  https://www.kaggle.com/api/v1/datasets/download/marcobeyer/bondora-p2p-loans

Execute:
    python -W default D00_DOMAIN_02_align_loan_vintage.py
"""

from __future__ import annotations

import json
import shutil
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


# Columns useful for survival + auction-time Cox (paper Appendix A spirit).
# Intersection with dump is taken at runtime.
KEEP_COLS = [
    "LoanId",
    "LoanNumber",
    "NewCreditCustomer",
    "LoanApplicationStartedDate",
    "LoanDate",
    "ContractEndDate",
    "FirstPaymentDate",
    "MaturityDate_Original",
    "MaturityDate_Last",
    "VerificationType",
    "LanguageCode",
    "Age",
    "Gender",
    "Country",
    "AppliedAmount",
    "Amount",
    "Interest",
    "LoanDuration",
    "MonthlyPayment",
    "Education",
    "EmploymentDurationCurrentEmployer",
    "EmploymentStatus",
    "OccupationArea",
    "HomeOwnershipType",
    "IncomeFromPrincipalEmployer",
    "IncomeFromPension",
    "IncomeFromFamilyAllowance",
    "IncomeFromSocialWelfare",
    "IncomeFromLeavePay",
    "IncomeFromChildSupport",
    "IncomeOther",
    "IncomeTotal",
    "ExistingLiabilities",
    "LiabilitiesTotal",
    "RefinanceLiabilities",
    "DebtToIncome",
    "FreeCash",
    "MonthlyPaymentDay",
    "ActiveScheduleFirstPaymentReached",
    "PlannedPrincipalTillDate",
    "PlannedInterestTillDate",
    "LastPaymentOn",
    "CurrentDebtDaysPrimary",
    "CurrentDebtDaysSecondary",
    "ExpectedLoss",
    "LossGivenDefault",
    "ExpectedReturn",
    "ProbabilityOfDefault",
    "DefaultDate",
    "Status",
    "Rating",
    "RescheduledOn",
    "PrincipalDebt",
    "InterestDebt",
    "PrincipalPaymentsMade",
    "InterestAndPenaltyPaymentsMade",
    "PrincipalWriteOffs",
    "InterestAndPenaltyWriteOffs",
    "PrincipalRecovery",
    "InterestAndPenaltyRecovery",
    "NoOfPreviousLoansBeforeLoan",
    "AmountOfPreviousLoansBeforeLoan",
    "PreviousRepaymentsBeforeLoan",
    "PreviousEarlyRepaymentsBefoleLoan",
    "PreviousEarlyRepaymentsBeforeLoan",
    "PreviousEarlyRepaymentsCountBeforeLoan",
    "GracePeriodStart",
    "GracePeriodEnd",
    "NextPaymentDate",
    "NextPaymentNr",
    "NrOfScheduledPayments",
    "ReScheduledOn",
    "ActiveLateCategory",
    "WorseLateCategory",
    "CreditScoreEsMicroL",
    "CreditScoreEsEquifaxRisk",
    "CreditScoreFiAsiakasTietoRiskGrade",
    "CreditScoreEeMini",
    "UseOfLoan",
    "ReportAsOfEOD",
]

# Explicitly drop even if present (target leakage / IDs / timing noise) — §3.2
DROP_ALWAYS = {
    "BidsPortfolioManager",
    "BidsApi",
    "BidsManual",
    "ListedOnUTC",
    "BiddingStartedOn",
    "ApplicationSignedHour",
    "ApplicationSignedWeekday",
    "UserName",
    "DateOfBirth",
    "City",
    "County",
}


def find_source() -> Path:
    """Prefer B01-installed canonical CSV, then known aliases under raw/bondora."""
    candidates = [
        cfg.DIRS["raw_bondora"] / cfg.BONDORA["loandata_local_name"],
        cfg.DIRS["raw_bondora"] / "LoanData_marcobeyer.csv",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 1_000_000:
            return p
    raise FileNotFoundError(
        "No LoanData.csv under data/raw/bondora/. "
        "Run B01_download_bondora.py first (Kaggle marcobeyer dump)."
    )


def parse_dates(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce")
    return out


def main() -> int:
    log("─" * 60)
    log("D00_DOMAIN_02 — ALIGN LOAN VINTAGE")
    log("─" * 60)

    src = find_source()
    log(f"  Source file: {src.relative_to(cfg.ROOT)} ({src.stat().st_size / 1e6:.1f} MB)")

    # Canonical raw copy under data/raw/bondora
    raw_dest = cfg.DIRS["raw_bondora"] / "LoanData.csv"
    cfg.DIRS["raw_bondora"].mkdir(parents=True, exist_ok=True)
    if src.resolve() != raw_dest.resolve():
        shutil.copy2(src, raw_dest)
        log(f"  Installed → {raw_dest.relative_to(cfg.ROOT)}")
    else:
        log(f"  Already at {raw_dest.relative_to(cfg.ROOT)}")

    df = pd.read_csv(raw_dest, low_memory=False)
    n0, c0 = len(df), len(df.columns)
    log(f"  Raw: rows={n0:,}  cols={c0}")

    retrieve = pd.Timestamp(cfg.DOMAIN_02["paper_retrieve_date"])
    loan_min = pd.Timestamp(cfg.DOMAIN_02["loan_date_min"])
    test_end = pd.Timestamp(cfg.DOMAIN_02["test_year_end"])  # 2020-12-31
    dur = int(cfg.DOMAIN_02["loan_duration_months"])

    date_cols = [
        "LoanDate",
        "DefaultDate",
        "ContractEndDate",
        "MaturityDate_Original",
        "MaturityDate_Last",
        "LastPaymentOn",
        "LoanApplicationStartedDate",
        "FirstPaymentDate",
        "ReportAsOfEOD",
    ]
    df = parse_dates(df, date_cols)

    # --- row filters (paper cohort window) ---
    if "LoanDuration" not in df.columns:
        log("ERROR: LoanDuration missing — wrong schema")
        return 1

    m_dur = df["LoanDuration"] == dur
    m_from = df["LoanDate"] >= loan_min
    m_to = df["LoanDate"] <= test_end  # exclude post-2020 (not in paper test/train)
    m_ok = df["LoanDate"].notna()
    before = len(df)
    df = df.loc[m_dur & m_from & m_to & m_ok].copy()
    log(
        f"  Row filter LoanDuration=={dur} & "
        f"{loan_min.date()}≤LoanDate≤{test_end.date()}: {before:,} → {len(df):,}"
    )

    # --- as-of retrieve date: hide future outcomes ---
    for c in ("DefaultDate", "ContractEndDate", "LastPaymentOn", "MaturityDate_Last"):
        if c in df.columns:
            future = df[c].notna() & (df[c] > retrieve)
            n_fut = int(future.sum())
            if n_fut:
                df.loc[future, c] = pd.NaT
                log(f"  Nullified {n_fut:,} future {c} (> {retrieve.date()})")

    df["ReportAsOfEOD"] = retrieve
    df["aligned_retrieve_date"] = retrieve

    # --- column filter ---
    drop_hit = [c for c in DROP_ALWAYS if c in df.columns]
    keep = [c for c in KEEP_COLS if c in df.columns]
    # always keep alignment meta
    for extra in ("aligned_retrieve_date", "ReportAsOfEOD"):
        if extra not in keep and extra in df.columns:
            keep.append(extra)
    # drop leakage-ish if still in keep (Status/DefaultDate kept for label construction in D01)
    keep = [c for c in keep if c not in drop_hit]
    missing_wanted = [c for c in KEEP_COLS if c not in df.columns]
    df_out = df[keep].copy()
    log(f"  Cols kept: {len(keep)} (dropped {c0 - len(keep)} from raw schema)")
    if missing_wanted:
        log(f"  Wanted but absent in dump ({len(missing_wanted)}): {missing_wanted[:12]}…")

    # --- write ---
    interim = cfg.DIRS["interim_d2"]
    interim.mkdir(parents=True, exist_ok=True)
    parquet = interim / "LoanData_aligned.parquet"
    csv_gz = interim / "LoanData_aligned.csv.gz"
    df_out.to_parquet(parquet, index=False)
    df_out.to_csv(csv_gz, index=False, compression="gzip")

    n_2020 = int(((df_out["LoanDate"] >= "2020-01-01") & (df_out["LoanDate"] <= test_end)).sum())
    n_pre = int((df_out["LoanDate"] < "2020-01-01").sum())

    report = {
        "stage": "D00_DOMAIN_02_align",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "kaggle_download_api": cfg.DATA_URLS["bondora_loandata_kaggle"],
            "kaggle_dataset": cfg.DATA_URLS["bondora_loandata_kaggle_slug"],
            "local_raw": str(raw_dest.relative_to(cfg.ROOT)),
            "input_path": str(src.relative_to(cfg.ROOT)),
            "bytes": raw_dest.stat().st_size,
        },
        "paper_retrieve_date": str(retrieve.date()),
        "filters": {
            "loan_duration_months": dur,
            "loan_date_min": str(loan_min.date()),
            "loan_date_max": str(test_end.date()),
            "test_year": "2020",
            "note": (
                "Rows restricted to 2014–2020 originations (36m) to mimic paper "
                "train+2020-test cohort; outcomes after retrieve date nullified."
            ),
        },
        "n_raw_rows": n0,
        "n_raw_cols": c0,
        "n_aligned_rows": int(len(df_out)),
        "n_aligned_cols": int(len(df_out.columns)),
        "n_pre_2020": n_pre,
        "n_2020_test_candidates": n_2020,
        "loan_date_min_aligned": str(df_out["LoanDate"].min().date()),
        "loan_date_max_aligned": str(df_out["LoanDate"].max().date()),
        "columns_kept": keep,
        "columns_missing_from_dump": missing_wanted,
        "columns_dropped_leakage": drop_hit,
        "outputs": {
            "parquet": str(parquet.relative_to(cfg.ROOT)),
            "csv_gz": str(csv_gz.relative_to(cfg.ROOT)),
        },
    }
    rep_path = cfg.DIRS["logs"] / "d00_domain_02_align.json"
    rep_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Provenance sidecar next to raw file
    (cfg.DIRS["raw_bondora"] / "SOURCE.md").write_text(
        "\n".join(
            [
                "# Bondora LoanData — provenance",
                "",
                f"- **Used for DOMAIN_02:** `{raw_dest.name}`",
                f"- **Kaggle API download:** {cfg.DATA_URLS['bondora_loandata_kaggle']}",
                f"- **Dataset slug:** `{cfg.DATA_URLS['bondora_loandata_kaggle_slug']}`",
                f"- **Paper retrieve date mimicked:** {retrieve.date()}",
                f"- **Aligned artifact:** `{parquet.relative_to(cfg.ROOT)}`",
                f"- **Logged:** `{rep_path.relative_to(cfg.ROOT)}`",
                "",
                "Investor portal `loan_dataset_investor.xlsx` is a different schema and is not used for Cox.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    log(f"  Aligned rows={len(df_out):,}  (pre-2020={n_pre:,}, 2020={n_2020:,})")
    log(f"  Wrote {parquet.relative_to(cfg.ROOT)}")
    log(f"  Report: {rep_path.relative_to(cfg.ROOT)}")
    log("D00_DOMAIN_02_align complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
