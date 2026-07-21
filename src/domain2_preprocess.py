"""
domain2_preprocess.py — Bone-Winkel & Reichenbach §3.2 preprocess (best-effort)
==============================================================================
Implements the paper's 10-step preprocessing spirit for DOMAIN_02:

  1. Drop post-auction / leakage columns
  2. Drop GDPR-era private fields when present
  3. Drop IDs / listing-timing noise
  4. Engineer DebtToIncomeModeled, RepaymentRatio + NaN flags
  5. Drop high-missing external credit scores
  6. Drop rows with any remaining missing in model fields (~2.85% in paper)
  7. Drop / collapse rare categories (<0.1%)
  8. Merge VerificationType unverified variants; build Country_Lang
  9. Rename categoricals to readable strings; NewBondoraCustomer
 10. One-hot encode categoricals (drop_first) for linear Cox / shared design

Feature set follows Table 4 *auction-time* Cox (excludes Bondora Rating /
ExpectedLoss / PD / LGD — those leak the platform score we compare against).

Not bit-identical without author code; deviations are logged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# LanguageCode mapping (Appendix A)
LANG = {
    1: "Estonian",
    2: "English",
    3: "Russian",
    4: "Finnish",
    5: "German",
    6: "Spanish",
    9: "Slovakian",
}

EDU = {
    1: "Primary",
    2: "Basic",
    3: "Vocational",
    4: "Secondary",
    5: "Higher",
}

GENDER = {0: "male", 1: "female", 2: "undefined"}

HOME = {
    0: "Homeless",
    1: "Owner",
    2: "LivingWithParents",
    3: "TenantFurnished",
    4: "TenantNotFurnished",
    5: "CouncilHouse",
    6: "JointTenant",
    7: "JointOwnership",
    8: "Mortgage",
    9: "OwnerEncumbrance",
    10: "Other",
}

VERIFY = {
    0: "not_set",
    1: "income_unverified",
    2: "income_unverified_phone",
    3: "income",
    4: "income_and_expenses",
}

LEAKAGE_COLS = [
    "Status",
    "DefaultDate",
    "ContractEndDate",
    "MaturityDate_Last",
    "LastPaymentOn",
    "PrincipalPaymentsMade",
    "InterestAndPenaltyPaymentsMade",
    "PrincipalWriteOffs",
    "InterestAndPenaltyWriteOffs",
    "PrincipalRecovery",
    "InterestAndPenaltyRecovery",
    "CurrentDebtDaysPrimary",
    "CurrentDebtDaysSecondary",
    "ActiveLateCategory",
    "WorseLateCategory",
    "GracePeriodStart",
    "GracePeriodEnd",
    "NextPaymentDate",
    "NextPaymentNr",
    "NrOfScheduledPayments",
    "ReScheduledOn",
    "PlannedPrincipalTillDate",
    "PlannedInterestTillDate",
    "ActiveScheduleFirstPaymentReached",
    # Bondora rating-model outputs (not used in fair Cox_2-style model)
    "ExpectedLoss",
    "ExpectedReturn",
    "ProbabilityOfDefault",
    "LossGivenDefault",
    "Rating",
]

GDPR_OR_PRIVATE = [
    "DateOfBirth",
    "City",
    "County",
    "UserName",
    "MaritalStatus",
    "NrOfDependants",
    "EmploymentStatus",
    "EmploymentPosition",
]

ID_TIMING = [
    "LoanId",
    "LoanNumber",
    "ListedOnUTC",
    "BiddingStartedOn",
    "ApplicationSignedHour",
    "ApplicationSignedWeekday",
    "MonthlyPaymentDay",
    "LoanApplicationStartedDate",
    "FirstPaymentDate",
    "BidsPortfolioManager",
    "BidsApi",
    "BidsManual",
]

CREDIT_SCORES_HIGH_MISS = [
    "CreditScoreEsMicroL",
    "CreditScoreEsEquifaxRisk",
    "CreditScoreFiAsiakasTietoRiskGrade",
    "CreditScoreEeMini",
]

# Numeric / engineered covariates (pre-dummies)
NUMERIC_MODEL = [
    "Age",
    "IncomeTotal",
    "DebtToIncomeModeled",
    "LiabilitiesTotal",
    "ExistingLiabilities",
    "AppliedAmount",
    "Interest",
    "AmountOfPreviousLoansBeforeLoan",
    "PreviousRepaymentsBeforeLoan",
    "RepaymentRatio",
    "PreviousEarlyRepaymentsBeforeLoan",
    "NoOfPreviousLoansBeforeLoan",
    "PreviousEarlyRepaymentsCountBeforeLoan",
]

CATEGORICAL_MODEL = [
    "Country_Lang",
    "Gender",
    "Education",
    "EmploymentDurationCurrentEmployer",
    "HomeOwnershipType",
    "VerificationType",
    "NewBondoraCustomer",
    "NanRepaymentHistory",
    "NanEarlyRepayment",
]

# Keep for labels / split / Bondora comparison (not Cox covariates)
META_KEEP = [
    "LoanId",
    "LoanDate",
    "LoanDuration",
    "duration_days",
    "event",
    "early_repayment",
    "DefaultDate",
    "ContractEndDate",
    "LastPaymentOn",  # Dömötör completed (≥1y without payment)
    "Status",
    "Rating",  # platform rating for Table 1 comparison only
    "Interest",  # also a covariate; kept in meta for rate tables
    "obs_end",
]


def _map_codes(series: pd.Series, mapping: dict) -> pd.Series:
    s = series.copy()
    # already strings?
    if s.dtype == object or str(s.dtype).startswith("string"):
        return s.astype(str)
    return s.map(mapping).fillna(s.astype(str))


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # DebtToIncomeModeled ≈ (LiabilitiesTotal + MonthlyPayment) / IncomeTotal
    inc = pd.to_numeric(out.get("IncomeTotal"), errors="coerce")
    liab = pd.to_numeric(out.get("LiabilitiesTotal"), errors="coerce").fillna(0.0)
    pay = pd.to_numeric(out.get("MonthlyPayment"), errors="coerce").fillna(0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        dti = (liab + pay) / inc.replace(0, np.nan)
    out["DebtToIncomeModeled"] = dti.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # RepaymentRatio + NaN flags (§3.2 footnotes 6–7)
    prev_rep = pd.to_numeric(out.get("PreviousRepaymentsBeforeLoan"), errors="coerce")
    prev_amt = pd.to_numeric(out.get("AmountOfPreviousLoansBeforeLoan"), errors="coerce")
    early_amt = pd.to_numeric(
        out.get("PreviousEarlyRepaymentsBeforeLoan"), errors="coerce"
    )
    if "PreviousEarlyRepaymentsBefoleLoan" in out.columns and early_amt.isna().all():
        early_amt = pd.to_numeric(out["PreviousEarlyRepaymentsBefoleLoan"], errors="coerce")

    nan_hist = prev_rep.isna() | prev_amt.isna() | (prev_amt.fillna(0) == 0)
    nan_early = early_amt.isna()
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = prev_rep / prev_amt.replace(0, np.nan)
    ratio = ratio.where(~nan_hist, 0.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    out["RepaymentRatio"] = ratio
    out["NanRepaymentHistory"] = nan_hist.map({True: "True", False: "False"})
    out["NanEarlyRepayment"] = nan_early.map({True: "True", False: "False"})
    out["PreviousEarlyRepaymentsBeforeLoan"] = early_amt.fillna(0.0)
    out["PreviousRepaymentsBeforeLoan"] = prev_rep.fillna(0.0)
    out["AmountOfPreviousLoansBeforeLoan"] = prev_amt.fillna(0.0)

    # NewBondoraCustomer
    ncc = out.get("NewCreditCustomer")
    if ncc is not None:
        if ncc.dtype == bool:
            out["NewBondoraCustomer"] = ncc.map({True: "True", False: "False"})
        else:
            out["NewBondoraCustomer"] = (
                pd.to_numeric(ncc, errors="coerce").fillna(0).astype(int).map({1: "True", 0: "False"})
            )
    else:
        out["NewBondoraCustomer"] = "False"

    # Categorical renames
    if "Gender" in out.columns:
        out["Gender"] = _map_codes(pd.to_numeric(out["Gender"], errors="coerce"), GENDER)
    if "Education" in out.columns:
        out["Education"] = _map_codes(pd.to_numeric(out["Education"], errors="coerce"), EDU)
    if "HomeOwnershipType" in out.columns:
        out["HomeOwnershipType"] = _map_codes(
            pd.to_numeric(out["HomeOwnershipType"], errors="coerce"), HOME
        )
    if "VerificationType" in out.columns:
        vt = _map_codes(pd.to_numeric(out["VerificationType"], errors="coerce"), VERIFY)
        # merge unverified variants (§3.2)
        vt = vt.replace(
            {
                "income_unverified_phone": "income_unverified",
                "not_set": "income_unverified",
            }
        )
        out["VerificationType"] = vt

    # Country_Lang: EE+Russian → EE_Ru; else Country
    country = out.get("Country")
    lang = out.get("LanguageCode")
    if country is not None:
        c = country.astype(str).str.upper()
        lang_name = _map_codes(pd.to_numeric(lang, errors="coerce"), LANG) if lang is not None else None
        cl = c.copy()
        if lang_name is not None:
            ee_ru = (c == "EE") & (lang_name == "Russian")
            cl = cl.where(~ee_ru, "EE_Ru")
        # drop rare SK (§3.2 ~41 Slovakian)
        out["Country_Lang"] = cl
    else:
        out["Country_Lang"] = "UNK"

    if "EmploymentDurationCurrentEmployer" in out.columns:
        out["EmploymentDurationCurrentEmployer"] = (
            out["EmploymentDurationCurrentEmployer"].astype(str).replace({"nan": np.nan})
        )

    return out


def drop_rare_categories(df: pd.DataFrame, min_frac: float = 0.001) -> pd.DataFrame:
    """Remove rows in categorical levels with share < min_frac (paper <0.1%)."""
    out = df.copy()
    n = len(out)
    if n == 0:
        return out
    mask = pd.Series(True, index=out.index)
    for col in ("Country_Lang", "HomeOwnershipType", "Education", "Gender"):
        if col not in out.columns:
            continue
        freq = out[col].value_counts(dropna=False) / n
        rare = freq[freq < min_frac].index
        if len(rare):
            mask &= ~out[col].isin(rare)
    # explicit paper drops
    if "Country_Lang" in out.columns:
        mask &= ~out["Country_Lang"].isin(["SK", "SK_Ru"])
    if "HomeOwnershipType" in out.columns:
        mask &= out["HomeOwnershipType"] != "Homeless"
    return out.loc[mask].copy()


def one_hot_design(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return dataframe with numeric + one-hot cols (drop_first) and feature name list."""
    pieces = [df[NUMERIC_MODEL].apply(pd.to_numeric, errors="coerce")]
    cat_present = [c for c in CATEGORICAL_MODEL if c in df.columns]
    dummies = pd.get_dummies(df[cat_present], columns=cat_present, drop_first=True, dtype=float)
    # sanitize column names for lifelines
    dummies.columns = [
        str(c)
        .replace(" ", "_")
        .replace(",", "")
        .replace("/", "_")
        .replace("-", "_")
        for c in dummies.columns
    ]
    X = pd.concat([pieces[0], dummies], axis=1)
    X = X.replace([np.inf, -np.inf], np.nan)
    feats = list(X.columns)
    return X, feats


def preprocess_for_modeling(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Full §3.2-style pipeline on a survival-labelled frame.

    Returns
    -------
    out : DataFrame
        Meta columns + one-hot design matrix columns.
    report : dict
        Counts / decisions for logging.
    """
    n0 = len(df)
    eng = engineer(df)

    # Rows: rare categories
    eng2 = drop_rare_categories(eng)
    n_rare = n0 - len(eng2)

    # Design matrix
    X, feats = one_hot_design(eng2)

    # Drop rows with any missing among model features (paper ~2.85%)
    complete = X.notna().all(axis=1)
    # also require duration/event
    complete &= eng2["duration_days"].notna() & eng2["event"].notna()
    n_miss = int((~complete).sum())

    meta_cols = [c for c in META_KEEP if c in eng2.columns]
    out = pd.concat([eng2.loc[complete, meta_cols], X.loc[complete]], axis=1)
    # avoid duplicate Interest if in both
    out = out.loc[:, ~out.columns.duplicated()]

    report = {
        "n_input": n0,
        "n_after_rare_filter": int(len(eng2)),
        "n_dropped_rare": int(n_rare),
        "n_dropped_missing": n_miss,
        "n_complete": int(len(out)),
        "n_features_design": len(feats),
        "feature_names": feats,
        "numeric_features": NUMERIC_MODEL,
        "categorical_features": [c for c in CATEGORICAL_MODEL if c in eng.columns],
        "excluded_bondora_score_cols": [
            c for c in LEAKAGE_COLS if c in ("ExpectedLoss", "Rating", "ProbabilityOfDefault")
        ],
        "note": (
            "Auction-time Cox design (Table 4 style without Bondora Rating/EL/PD/LGD). "
            "Not bit-identical to author code."
        ),
    }
    return out, report
