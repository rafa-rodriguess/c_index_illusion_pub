"""
D00_DOMAIN_03_features_stackexchange.py — URV features + survival labels
=======================================================================
Lane: DOMAIN_03 (Abedi Firouzjaei 2022).

Reads author ``user_features.csv`` (tab-separated) for Pol/DS/CS and builds
survival tables for θ ∈ {24, 36} months following the author RSF notebook +
paper §5.2 contributor filter:

  event = 1[MonthsSinceLastActivity > θ]
  duration = MonthsActive
  keep users in Q ∪ A ∪ C ∪ U ∪ D (any contribution count > 0)

Execute:
    python -W default D00_DOMAIN_03_features_stackexchange.py
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


def feature_cols(name: str) -> list[str]:
    fs = cfg.DOMAIN_03["feature_sets"]
    if name == "combined":
        return list(fs["behavioural"]) + list(fs["content"])
    return list(fs[name])


def load_community(code: str) -> pd.DataFrame:
    path = cfg.DIRS["raw_stackexchange"] / "processed_data" / code / "user_features.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path} — run B02_download_stackexchange.py")
    return pd.read_csv(path, sep="\t")


def contributor_mask(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in cfg.DOMAIN_03["contributor_cols"] if c in df.columns]
    if not cols:
        raise ValueError("No contributor columns present in user_features")
    m = pd.Series(False, index=df.index)
    for c in cols:
        m = m | (pd.to_numeric(df[c], errors="coerce").fillna(0) > 0)
    return m


def build_survival(df: pd.DataFrame, theta: int) -> pd.DataFrame:
    out = df.copy()
    out["theta"] = theta
    out["disengaged"] = (out[cfg.DOMAIN_03["inactivity_col"]] > theta).astype(int)
    dur = pd.to_numeric(out[cfg.DOMAIN_03["duration_col"]], errors="coerce").fillna(0)
    # PySurvival / RSF need non-negative times; clip tiny zeros only
    out["duration_months"] = dur.clip(lower=1e-3)
    out["event"] = out["disengaged"]
    return out


def main() -> int:
    log("─" * 60)
    log("D00_DOMAIN_03 — FEATURES / SURVIVAL LABELS")
    log("─" * 60)

    out_dir = cfg.DIRS["processed_d3"]
    out_dir.mkdir(parents=True, exist_ok=True)
    use_filter = bool(cfg.DOMAIN_03.get("contributor_filter", True))

    summary = {
        "stage": "D00_DOMAIN_03",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "github": cfg.DOMAIN_03["github"],
        "contributor_filter": use_filter,
        "contributor_cols": list(cfg.DOMAIN_03["contributor_cols"]),
        "feature_sets": {
            "behavioural": feature_cols("behavioural"),
            "content": feature_cols("content"),
            "combined": feature_cols("combined"),
        },
        "communities": {},
        "decisions": [
            {
                "id": "D00.1",
                "choice": "Use author processed user_features.csv (tab-separated)",
                "source": "habedi/SurvivalAnalysisQACommunities",
            },
            {
                "id": "D00.2",
                "choice": "event = MonthsSinceLastActivity > θ; duration = MonthsActive",
                "source": "paper §4.6 + author RSF notebook",
            },
            {
                "id": "D00.3",
                "choice": (
                    "Contributor filter Q∪A∪C∪U∪D = any of "
                    f"{list(cfg.DOMAIN_03['contributor_cols'])} > 0"
                    if use_filter
                    else "No contributor pre-filter (notebook-only)"
                ),
                "source": "paper §5.2 / Table 6",
            },
        ],
    }

    for code in cfg.DOMAIN_03["communities"]:
        raw = load_community(code)
        label = cfg.DOMAIN_03["community_labels"][code]
        n_raw = len(raw)
        if use_filter:
            keep = contributor_mask(raw)
            filtered = raw.loc[keep].reset_index(drop=True)
        else:
            filtered = raw
        log(
            f"  {label} ({code}): raw={n_raw:,}  "
            f"after_filter={len(filtered):,}  "
            f"kept={100 * len(filtered) / max(n_raw, 1):.1f}%"
        )
        comm = {
            "n_raw": int(n_raw),
            "n_after_filter": int(len(filtered)),
            "theta": {},
        }
        for theta in cfg.DOMAIN_03["theta_months"]:
            surv = build_survival(filtered, int(theta))
            path = out_dir / f"{code}_theta{theta}.parquet"
            keep_cols = (
                [
                    "UserId",
                    "duration_months",
                    "event",
                    "theta",
                    "MonthsSinceLastActivity",
                    "MonthsActive",
                ]
                + feature_cols("combined")
            )
            keep_cols = [c for c in keep_cols if c in surv.columns]
            surv[keep_cols].to_parquet(path, index=False)
            n_evt = int(surv["event"].sum())
            comm["theta"][str(theta)] = {
                "path": str(path.relative_to(cfg.ROOT)),
                "n": int(len(surv)),
                "n_events": n_evt,
                "event_rate": float(n_evt / len(surv)) if len(surv) else 0.0,
            }
            log(
                f"    θ={theta}: n={len(surv):,}  events={n_evt:,}  "
                f"rate={n_evt / max(len(surv), 1):.3f} → {path.name}"
            )
        summary["communities"][code] = comm

    dec = cfg.DIRS["logs"] / "d00_domain_03_decisions.json"
    dec.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(f"  Decisions: {dec.relative_to(cfg.ROOT)}")
    log("D00_DOMAIN_03 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
