"""
D00_DOMAIN_01_features_backblaze.py — Domain 1 feature / survival table
=======================================================================
Lane: DOMAIN_01 (Ahmed & Green 2024).

Builds the drive-level survival table needed to reproduce the Cox PH
C-index of 0.958 from Backblaze Drive Stats.

Paper protocol (cfg.DOMAIN_01 + §§4–7.1):
  - Model filter: ST4000DM000
  - Window: 2013-01-01 .. 2022-12-31
  - 21 SMART raw attrs (omit 3, 10, 191)
  - Reported population: 37,037 drives
  - Cox GOF cohort (H6a): healthy with calendar span > 7y UNION all failed
    (paper §4.1 reported 12,993 healthy + 4,889 failed; smoke C≈0.958)
  - Failure horizon 15 days applies to the DL labelling path; Cox uses
    actual failure / right-censor times (documented in decisions log)

Author code: GitLab https://gitlab.com/Jishan/deeplearning2023 → URL 404
even when authenticated (2026-07-12). See domain1-ahmed-green-2024/CODE_ACCESS.md.

Execute:
    python -W default D00_DOMAIN_01_features_backblaze.py
    python -W default D00_DOMAIN_01_features_backblaze.py --inventory-only
    python -W default D00_DOMAIN_01_features_backblaze.py --amend-existing

Done when:
    - data/processed/domain1/drives.parquet (+ .csv.gz fallback)
    - data/processed/domain1/drives_cox_cohort.parquet  (H6a slice)
    - results/logs/d00_domain_01_inventory.json
    - results/logs/d00_domain_01_decisions.json
    - exit 0 under python -W default
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import time
import warnings
import zipfile
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import cfg
from src.domain1_cox_cohort import cohort_count_summary, ensure_cohort_columns

warnings.filterwarnings("default")


def log(msg: str = "") -> None:
    print(msg, flush=True)


# ── Protocol constants from paper / cfg ──────────────────────────────────────

SMART_IDS: list[int] = list(cfg.DOMAIN_01["smart_ids"])
SMART_COLS: list[str] = [f"smart_{sid}_raw" for sid in SMART_IDS]
MODEL = str(cfg.DOMAIN_01["model_filter"])
HOURS_PER_YEAR = 365.25 * 24
MIN_AGE_YEARS = float(cfg.DOMAIN_01["cox_cohort_min_age_years"])
MIN_AGE_HOURS = MIN_AGE_YEARS * HOURS_PER_YEAR


def expected_smart_raw_cols() -> list[str]:
    return list(SMART_COLS)


def list_zips() -> list[Path]:
    raw = cfg.DIRS["raw_backblaze"]
    return [raw / name for name in cfg.BACKBLAZE["zip_filenames"] if (raw / name).exists()]


def is_data_csv(member: str) -> bool:
    if not member.lower().endswith(".csv"):
        return False
    name = Path(member).name
    if name.startswith("._") or member.startswith("__MACOSX"):
        return False
    return True


def peek_csv_header(zip_path: Path) -> tuple[str, list[str]]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = sorted(m for m in zf.namelist() if is_data_csv(m))
        if not members:
            raise RuntimeError(f"No CSV members in {zip_path.name}")
        member = members[0]
        with zf.open(member) as fh:
            text = io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
            header = next(csv.reader(text))
        return member, header


def sample_model_present(zip_path: Path, model: str, max_rows: int = 50_000) -> bool:
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = sorted(m for m in zf.namelist() if is_data_csv(m))
        if not members:
            return False
        with zf.open(members[0]) as fh:
            text = io.TextIOWrapper(fh, encoding="utf-8", errors="replace")
            reader = csv.DictReader(text)
            for i, row in enumerate(reader):
                if i >= max_rows:
                    break
                if row.get("model") == model:
                    return True
    return False


def build_inventory() -> dict[str, Any]:
    zips = list_zips()
    expected = set(cfg.BACKBLAZE["zip_filenames"])
    found = {p.name for p in zips}
    missing = sorted(expected - found)

    probe_zip = None
    for preferred in ("data_2015.zip", "data_Q1_2016.zip", "data_2013.zip"):
        cand = cfg.DIRS["raw_backblaze"] / preferred
        if cand.exists():
            probe_zip = cand
            break
    if probe_zip is None and zips:
        probe_zip = zips[0]

    schema: dict[str, Any] = {}
    model_hit = False
    if probe_zip is not None:
        member, header = peek_csv_header(probe_zip)
        smart_raw = expected_smart_raw_cols()
        present = [c for c in smart_raw if c in header]
        missing_cols = [c for c in smart_raw if c not in header]
        schema = {
            "probe_zip": probe_zip.name,
            "probe_member": member,
            "n_columns": len(header),
            "has_serial_number": "serial_number" in header,
            "has_model": "model" in header,
            "has_failure": "failure" in header,
            "smart_raw_present": len(present),
            "smart_raw_expected": len(smart_raw),
            "smart_raw_missing": missing_cols,
            "header_sample": header[:30],
        }
        log(f"  Probing model filter in {probe_zip.name} (up to 50k rows)…")
        model_hit = sample_model_present(probe_zip, MODEL)

    return {
        "stage": "D00_DOMAIN_01",
        "phase": "inventory",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "model_filter": MODEL,
            "period": [cfg.DOMAIN_01["period_start"], cfg.DOMAIN_01["period_end"]],
            "smart_ids": SMART_IDS,
            "omit_smart_ids": cfg.DOMAIN_01["omit_smart_ids"],
            "cox_cohort_min_age_years": MIN_AGE_YEARS,
            "target_cindex": cfg.DOMAIN_01["target_value"],
            "n_drives_reported": cfg.DOMAIN_01["n_drives_reported"],
            "cox_cohort_reported": cfg.DOMAIN_01["cox_cohort_reported"],
            "failure_horizon_days": cfg.DOMAIN_01["failure_horizon_days"],
            "author_code_url": "https://gitlab.com/Jishan/deeplearning2023",
            "author_code_status": "private_unavailable",
        },
        "raw_dir": str(cfg.DIRS["raw_backblaze"]),
        "zips_expected": len(expected),
        "zips_found": len(found),
        "zips_missing": missing,
        "schema": schema,
        "model_filter_seen_in_probe": model_hit,
        "status": (
            "inventory_ok"
            if not missing and schema.get("smart_raw_present", 0) >= 18
            else "inventory_partial"
        ),
    }


# ── Phase 1: drive-level aggregation ─────────────────────────────────────────


@dataclass
class DriveState:
    first_date: date
    last_date: date
    failed: int = 0
    n_rows: int = 0
    smart_last: dict[str, float] = field(default_factory=dict)


def parse_date(s: str) -> date:
    """Backblaze date formats vary by year (ISO vs US slash)."""
    s = (s or "").strip()
    if not s:
        raise ValueError("empty date")
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return date.fromisoformat(s[:10])
    for fmt in ("%m/%d/%y", "%m/%d/%Y", "%Y/%m/%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unrecognized date: {s!r}")


def parse_float(s: str | None) -> float | None:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def update_drive(state: dict[str, DriveState], row: dict[str, str]) -> None:
    sn = row.get("serial_number")
    if not sn:
        return
    d = parse_date(row["date"])
    fail = int(float(row.get("failure") or 0))
    st = state.get(sn)
    if st is None:
        st = DriveState(first_date=d, last_date=d)
        state[sn] = st
        _write_smart(st, row)
    else:
        if d < st.first_date:
            st.first_date = d
        if d > st.last_date:
            st.last_date = d
            _write_smart(st, row)
        elif d == st.last_date:
            _write_smart(st, row)
    st.n_rows += 1
    if fail:
        st.failed = 1


def _write_smart(st: DriveState, row: dict[str, str]) -> None:
    for col in SMART_COLS:
        if col not in row:
            continue
        val = parse_float(row.get(col))
        if val is not None:
            st.smart_last[col] = val


def process_zip(zip_path: Path, state: dict[str, DriveState]) -> dict[str, int]:
    n_rows = 0
    n_model = 0
    n_csv = 0
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = sorted(m for m in zf.namelist() if is_data_csv(m))
        for member in members:
            n_csv += 1
            with zf.open(member) as fh:
                text = io.TextIOWrapper(fh, encoding="utf-8", errors="replace", newline="")
                reader = csv.DictReader(text)
                for row in reader:
                    n_rows += 1
                    if row.get("model") != MODEL:
                        continue
                    n_model += 1
                    update_drive(state, row)
    return {"n_csv": n_csv, "n_rows": n_rows, "n_model_rows": n_model}


def checkpoint_path() -> Path:
    return cfg.DIRS["interim_d1"] / "d00_drive_state.joblib"


def save_checkpoint(state: dict[str, DriveState], done_zips: list[str]) -> None:
    import joblib

    path = checkpoint_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"state": state, "done_zips": done_zips}, path)


def load_checkpoint() -> tuple[dict[str, DriveState], list[str]]:
    import joblib

    path = checkpoint_path()
    if not path.exists():
        return {}, []
    blob = joblib.load(path)
    return blob["state"], list(blob["done_zips"])


def state_to_records(state: dict[str, DriveState]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for sn, st in state.items():
        duration_days = (st.last_date - st.first_date).days + 1
        calendar_span_years = duration_days / 365.25
        poh = st.smart_last.get("smart_9_raw")
        age_years = (poh / HOURS_PER_YEAR) if poh is not None else None
        event = int(st.failed)
        cox_cohort_smart9 = bool(age_years is not None and age_years > MIN_AGE_YEARS)
        # H6a: all failed ∪ healthy with calendar span > 7y
        cox_fit_h6a = bool(event == 1 or (event == 0 and calendar_span_years > MIN_AGE_YEARS))
        rec: dict[str, Any] = {
            "serial_number": sn,
            "first_date": st.first_date.isoformat(),
            "last_date": st.last_date.isoformat(),
            "duration_days": duration_days,
            "event": event,
            "n_daily_rows": st.n_rows,
            "power_on_hours": poh,
            "age_years_smart9": age_years,
            "calendar_span_years": calendar_span_years,
            "cox_cohort_age_gt7": cox_cohort_smart9,  # legacy SMART9 both-class flag
            "cox_fit_h6a": cox_fit_h6a,
        }
        for col in SMART_COLS:
            rec[col] = st.smart_last.get(col)
        records.append(rec)
    return records


def write_drives_table(records: list[dict[str, Any]]) -> dict[str, str]:
    import pandas as pd

    out_dir = cfg.DIRS["processed_d1"]
    out_dir.mkdir(parents=True, exist_ok=True)
    df = ensure_cohort_columns(pd.DataFrame.from_records(records))
    base_cols = [
        "serial_number",
        "first_date",
        "last_date",
        "duration_days",
        "event",
        "n_daily_rows",
        "power_on_hours",
        "age_years_smart9",
        "calendar_span_years",
        "cox_cohort_age_gt7",
        "cox_fit_h6a",
    ]
    df = df[base_cols + SMART_COLS]
    paths: dict[str, str] = {}
    parquet = out_dir / "drives.parquet"
    try:
        df.to_parquet(parquet, index=False)
        paths["parquet"] = str(parquet.relative_to(cfg.ROOT))
    except Exception as exc:  # noqa: BLE001 — fallback if engine missing
        log(f"  parquet write failed ({exc}); writing csv.gz only")
    csv_gz = out_dir / "drives.csv.gz"
    df.to_csv(csv_gz, index=False, compression="gzip")
    paths["csv_gz"] = str(csv_gz.relative_to(cfg.ROOT))
    # Cox fit slice = H6a
    cohort = df.loc[df["cox_fit_h6a"].astype(bool)].copy()
    cohort_path = out_dir / "drives_cox_cohort.parquet"
    try:
        cohort.to_parquet(cohort_path, index=False)
        paths["cox_cohort_parquet"] = str(cohort_path.relative_to(cfg.ROOT))
    except Exception:
        cohort_csv = out_dir / "drives_cox_cohort.csv.gz"
        cohort.to_csv(cohort_csv, index=False, compression="gzip")
        paths["cox_cohort_csv_gz"] = str(cohort_csv.relative_to(cfg.ROOT))
    return paths


def amend_existing_table() -> dict[str, Any]:
    """Recompute H6a flags on an existing drives table (no zip re-stream)."""
    import pandas as pd

    out_dir = cfg.DIRS["processed_d1"]
    src = out_dir / "drives.parquet"
    if not src.exists():
        raise FileNotFoundError(f"Missing {src}; run a full D00 first.")
    t0 = time.perf_counter()
    df = pd.read_parquet(src)
    framed = ensure_cohort_columns(df)
    records = framed.to_dict(orient="records")
    paths = write_drives_table(records)
    summary = cohort_count_summary(framed)
    summary["elapsed_s"] = round(time.perf_counter() - t0, 1)
    summary["outputs"] = paths
    summary["paper_n_drives"] = cfg.DOMAIN_01["n_drives_reported"]
    summary["delta_n_drives"] = summary["n_drives_total"] - cfg.DOMAIN_01["n_drives_reported"]
    summary["amend_only"] = True
    return summary


def decisions_document(stats: dict[str, Any]) -> dict[str, Any]:
    """Every non-trivial choice, tied to paper section or flagged as gap."""
    return {
        "stage": "D00_DOMAIN_01",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "author_code": {
            "url": cfg.DOMAIN_01.get("author_code_url"),
            "status": cfg.DOMAIN_01.get("author_code_status"),
            "note": (
                "Paper footnote promises GitLab; URL returns 404 even when authenticated "
                "(checked 2026-07-12). Reproduction follows published text + smoke H6a."
            ),
        },
        "decisions": [
            {
                "id": "D00.1",
                "choice": "Filter model == ST4000DM000",
                "source": "paper §4 (37,037 Seagate ST4000DM000, 2013–2022)",
            },
            {
                "id": "D00.2",
                "choice": "Use 21 SMART *_raw columns; omit SMART 3, 10, 191",
                "source": "paper §4 (V1–V21 mapping)",
            },
            {
                "id": "D00.3",
                "choice": (
                    "One row per serial_number; duration_days = last_date - first_date + 1; "
                    "event = 1 if any daily failure flag else 0 (right censor at study end / removal)"
                ),
                "source": "paper §3.3 (right censoring) + §4.1 survival framing",
            },
            {
                "id": "D00.4",
                "choice": "Covariates = SMART raw values on the drive's last observed day",
                "source": (
                    "paper §3.3.2 / §7.1 (Cox on SMART → survival time) does not specify "
                    "which snapshot; last-observation is the standard time-fixed summary "
                    "when code is unavailable — FLAGGED"
                ),
                "flag": "underspecified_in_paper",
            },
            {
                "id": "D00.5",
                "choice": (
                    "Cox fit cohort H6a: healthy with calendar_span_years > 7 "
                    "UNION all failed (flag cox_fit_h6a). Legacy cox_cohort_age_gt7 "
                    f"(SMART9 > {MIN_AGE_YEARS}y, both classes) retained for DL audits only."
                ),
                "source": (
                    "paper §4.1 counts (12,993 healthy + 4,889 failed) + smoke 2026-07-12: "
                    "H6a in-sample C≈0.9595 vs paper 0.958; SMART9-both-classes fails (~147 events)"
                ),
            },
            {
                "id": "D00.6",
                "choice": (
                    "15-day failure horizon NOT applied to Cox table "
                    "(horizon is for DL instance labelling in §4.1 / §6)"
                ),
                "source": "paper §4.1 failure horizon paragraph vs §7.1 Cox GOF",
            },
            {
                "id": "D00.7",
                "choice": "Process zips in cfg order (2013 annual → 2022 Q4); skip __MACOSX / ._ files",
                "source": "engineering; preserves chronological last-SMART updates",
            },
        ],
        "counts": stats,
        "paper_targets": {
            "n_drives": cfg.DOMAIN_01["n_drives_reported"],
            "cox_healthy": cfg.DOMAIN_01["cox_cohort_reported"]["healthy"],
            "cox_failed": cfg.DOMAIN_01["cox_cohort_reported"]["failed"],
        },
    }


def run_phase1() -> dict[str, Any]:
    zips = list_zips()
    if len(zips) != len(cfg.BACKBLAZE["zip_filenames"]):
        missing = sorted(
            set(cfg.BACKBLAZE["zip_filenames"]) - {p.name for p in zips}
        )
        raise FileNotFoundError(f"Missing Backblaze zips: {missing}")

    state, done_zips = load_checkpoint()
    if done_zips:
        log(f"  Resuming from checkpoint ({len(done_zips)} zips done, {len(state):,} drives)")

    per_zip: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    log(f"  Streaming {len(zips)} zips → drive-level state (model={MODEL})…")
    for i, zp in enumerate(zips, 1):
        if zp.name in done_zips:
            log(f"  [{i:02d}/{len(zips)}] {zp.name}: skipped (checkpoint)")
            continue
        t_zip = time.perf_counter()
        stats = process_zip(zp, state)
        stats["zip"] = zp.name
        stats["elapsed_s"] = round(time.perf_counter() - t_zip, 1)
        stats["unique_drives_so_far"] = len(state)
        per_zip.append(stats)
        done_zips.append(zp.name)
        save_checkpoint(state, done_zips)
        log(
            f"  [{i:02d}/{len(zips)}] {zp.name}: "
            f"model_rows={stats['n_model_rows']:,}  "
            f"drives={len(state):,}  ({stats['elapsed_s']}s)"
        )

    records = state_to_records(state)
    summary = cohort_count_summary(
        __import__("pandas").DataFrame.from_records(records)
    )
    paths = write_drives_table(records)
    elapsed = round(time.perf_counter() - t0, 1)

    cp = checkpoint_path()
    if cp.exists():
        cp.unlink()

    counts = {
        **summary,
        "paper_n_drives": cfg.DOMAIN_01["n_drives_reported"],
        "delta_n_drives": summary["n_drives_total"] - cfg.DOMAIN_01["n_drives_reported"],
        "elapsed_s": elapsed,
        "outputs": paths,
        "per_zip": per_zip,
        # aliases used by older D03 readers
        "n_cox_cohort": summary["n_cox_fit"],
    }
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory-only",
        action="store_true",
        help="Only write inventory JSON (Phase 0).",
    )
    parser.add_argument(
        "--amend-existing",
        action="store_true",
        help="Recompute H6a cohort flags on existing drives.parquet (no zip re-stream).",
    )
    args = parser.parse_args(argv)

    log("─" * 60)
    log("D00_DOMAIN_01 — FEATURES (Ahmed & Green)")
    log("─" * 60)
    log(f"  Raw dir : {cfg.DIRS['raw_backblaze']}")
    log(f"  Model   : {MODEL}")
    log(f"  SMART   : {len(SMART_IDS)} features")
    log(f"  Cox pop : {cfg.DOMAIN_01.get('cox_fit_population')}")
    log()

    inv = build_inventory()
    out_inv = cfg.DIRS["logs"] / "d00_domain_01_inventory.json"
    out_inv.parent.mkdir(parents=True, exist_ok=True)
    out_inv.write_text(json.dumps(inv, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(f"  Inventory: {inv['status']} → {out_inv.relative_to(cfg.ROOT)}")

    if args.inventory_only:
        log("D00_DOMAIN_01 complete (inventory only).")
        return 0 if inv["status"] in {"inventory_ok", "inventory_partial"} else 1

    if args.amend_existing:
        log("  Mode: --amend-existing (H6a flags only)")
        counts = amend_existing_table()
    else:
        if inv["status"] not in {"inventory_ok", "inventory_partial"}:
            log("ERROR: inventory failed; aborting Phase 1.")
            return 1
        log()
        counts = run_phase1()

    decisions = decisions_document(counts)
    out_dec = cfg.DIRS["logs"] / "d00_domain_01_decisions.json"
    out_dec.write_text(json.dumps(decisions, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    log()
    log("  Population vs paper:")
    log(
        f"    drives     ours={counts['n_drives_total']:,}  "
        f"paper={counts.get('paper_n_drives', cfg.DOMAIN_01['n_drives_reported']):,}  "
        f"Δ={counts.get('delta_n_drives', 0):+,}"
    )
    log(
        f"    H6a OK     ours={counts['n_cox_healthy']:,}  "
        f"paper={counts['paper_cox_healthy']:,}  Δ={counts['delta_cox_healthy']:+,}"
    )
    log(
        f"    H6a fail   ours={counts['n_cox_failed']:,}  "
        f"paper={counts['paper_cox_failed']:,}  Δ={counts['delta_cox_failed']:+,}"
    )
    log(f"  Outputs : {counts.get('outputs')}")
    log(f"  Decisions log: {out_dec.relative_to(cfg.ROOT)}")
    log(f"  Elapsed : {counts.get('elapsed_s')}s")
    log("D00_DOMAIN_01 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
