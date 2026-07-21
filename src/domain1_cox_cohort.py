"""
DOMAIN_01 Cox fit population helpers (Ahmed & Green).

H6a (locked 2026-07-12 smoke): calendar span > 7y **healthy** UNION **all failed**.
That population yields in-sample Harrell C ≈ paper 0.958.
"""

from __future__ import annotations

import pandas as pd

from src.config import cfg

HOURS_PER_YEAR = 365.25 * 24.0


def ensure_cohort_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with calendar_span_years + cox_fit_h6a (+ legacy SMART9 flag if missing).

    ``calendar_span_years`` matches D00: inclusive observation span =
    ``duration_days / 365.25`` when present, else ``((last−first).days + 1) / 365.25``.
    """
    out = df.copy()
    min_age = float(cfg.DOMAIN_01["cox_cohort_min_age_years"])

    if "duration_days" in out.columns:
        out["calendar_span_years"] = out["duration_days"].astype(float) / 365.25
    else:
        fd = pd.to_datetime(out["first_date"])
        ld = pd.to_datetime(out["last_date"])
        out["calendar_span_years"] = ((ld - fd).dt.days + 1) / 365.25

    if "age_years_smart9" not in out.columns and "power_on_hours" in out.columns:
        poh = out["power_on_hours"]
        out["age_years_smart9"] = poh / HOURS_PER_YEAR

    if "cox_cohort_age_gt7" not in out.columns:
        age = out.get("age_years_smart9")
        if age is not None:
            out["cox_cohort_age_gt7"] = age.notna() & (age > min_age)
        else:
            out["cox_cohort_age_gt7"] = False

    # H6a: all failures ∪ healthy with calendar age > 7y
    out["cox_fit_h6a"] = (out["event"].astype(int) == 1) | (
        (out["event"].astype(int) == 0) & (out["calendar_span_years"] > min_age)
    )
    return out


def select_cox_fit_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Filter drives to the configured Cox GOF population."""
    pop = str(cfg.DOMAIN_01.get("cox_fit_population", ""))
    framed = ensure_cohort_columns(df)
    if pop in {
        "h6a_calgt7_healthy_union_all_failed",
        "asymmetric_calgt7_healthy_union_all_failed",
    }:
        return framed.loc[framed["cox_fit_h6a"].astype(bool)].copy()
    if pop == "all_drives_complete_cases":
        return framed
    # Unknown key → fail loud rather than silently using all drives
    raise ValueError(
        f"Unknown DOMAIN_01 cox_fit_population={pop!r}. "
        "Expected h6a_calgt7_healthy_union_all_failed or all_drives_complete_cases."
    )


def cohort_count_summary(df: pd.DataFrame) -> dict:
    """Counts for H6a vs legacy SMART9 both-class flag vs paper §4.1."""
    framed = ensure_cohort_columns(df)
    h6a = framed.loc[framed["cox_fit_h6a"].astype(bool)]
    smart9 = framed.loc[framed["cox_cohort_age_gt7"].astype(bool)]
    paper = cfg.DOMAIN_01["cox_cohort_reported"]
    return {
        "n_drives_total": int(len(framed)),
        "n_healthy": int((framed["event"] == 0).sum()),
        "n_failed": int((framed["event"] == 1).sum()),
        "n_cox_fit": int(len(h6a)),
        "n_cox_healthy": int((h6a["event"] == 0).sum()),
        "n_cox_failed": int((h6a["event"] == 1).sum()),
        "n_smart9_both_gt7": int(len(smart9)),
        "n_smart9_gt7_healthy": int((smart9["event"] == 0).sum()),
        "n_smart9_gt7_failed": int((smart9["event"] == 1).sum()),
        "paper_cox_healthy": int(paper["healthy"]),
        "paper_cox_failed": int(paper["failed"]),
        "delta_cox_healthy": int((h6a["event"] == 0).sum()) - int(paper["healthy"]),
        "delta_cox_failed": int((h6a["event"] == 1).sum()) - int(paper["failed"]),
        "cox_fit_population": cfg.DOMAIN_01.get("cox_fit_population"),
    }
