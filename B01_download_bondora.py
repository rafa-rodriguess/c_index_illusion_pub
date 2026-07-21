"""
B01_download_bondora.py — Acquire Bondora LoanData for DOMAIN_02
================================================================
Canonical raw input for Bone-Winkel & Reichenbach reproduction:

    data/raw/bondora/LoanData.csv

Source (logged):
    https://www.kaggle.com/api/v1/datasets/download/marcobeyer/bondora-p2p-loans
    dataset slug: marcobeyer/bondora-p2p-loans

Acquisition order (first success wins):
  1. Existing ``LoanData.csv`` matching pinned SHA-256 (reproducibility gate)
  2. Download via Kaggle API (``~/.kaggle/kaggle.json`` or env
     ``KAGGLE_USERNAME`` / ``KAGGLE_KEY``), unzip, place ``LoanData.csv``

Also (non-blocking):
  - optional download of portal ``loan_dataset_investor.xlsx`` (wrong schema; legacy)
  - probe legacy Bondora zip URLs (403 expected)

Exit 0 iff canonical ``LoanData.csv`` is present and hash-verified.

Execute:
    python -W default B01_download_bondora.py
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import sys
import warnings
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests

from src.config import cfg

warnings.filterwarnings("default")


def log(msg: str = "") -> None:
    print(msg, flush=True)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def probe_url(url: str, timeout: int) -> dict:
    try:
        with requests.head(url, timeout=timeout, allow_redirects=True) as resp:
            return {
                "url": url,
                "http_status": resp.status_code,
                "available": resp.status_code == 200,
                "content_length": resp.headers.get("Content-Length"),
            }
    except requests.RequestException as exc:
        return {
            "url": url,
            "http_status": None,
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def download_file(url: str, dest: Path, timeout: int, chunk: int) -> str:
    """Best-effort GET for optional xlsx; return skipped|downloaded|failed."""
    try:
        expected = None
        with requests.head(url, timeout=timeout, allow_redirects=True) as resp:
            if resp.ok and resp.headers.get("Content-Length"):
                expected = int(resp.headers["Content-Length"])
        if dest.exists() and expected is not None and dest.stat().st_size == expected:
            return "skipped"
        if dest.exists() and expected is None and dest.stat().st_size > 0:
            return "skipped"
        with requests.get(url, stream=True, timeout=timeout, allow_redirects=True) as resp:
            resp.raise_for_status()
            with dest.open("wb") as fh:
                for block in resp.iter_content(chunk_size=chunk):
                    if block:
                        fh.write(block)
        return "downloaded"
    except Exception as exc:  # noqa: BLE001
        log(f"  optional xlsx download failed: {type(exc).__name__}: {exc}")
        return "failed"


def kaggle_credentials() -> tuple[str, str] | None:
    user = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if user and key:
        return user, key
    cred_path = Path.home() / ".kaggle" / "kaggle.json"
    if cred_path.exists():
        data = json.loads(cred_path.read_text(encoding="utf-8"))
        if data.get("username") and data.get("key"):
            return str(data["username"]), str(data["key"])
    return None


def download_kaggle_loandata(dest_csv: Path, timeout: int) -> str:
    """Download marcobeyer zip via Kaggle API and extract LoanData.csv."""
    creds = kaggle_credentials()
    if creds is None:
        raise RuntimeError(
            "Kaggle credentials not found. Place ~/.kaggle/kaggle.json or set "
            "KAGGLE_USERNAME / KAGGLE_KEY."
        )
    user, key = creds
    url = cfg.DATA_URLS["bondora_loandata_kaggle"]
    log(f"  Kaggle API GET {url}")
    with requests.get(
        url,
        auth=(user, key),
        stream=True,
        timeout=timeout,
        allow_redirects=True,
    ) as resp:
        resp.raise_for_status()
        blob = resp.content if False else b"".join(resp.iter_content(chunk_size=1 << 20))

    # Response is a zip archive
    zf = zipfile.ZipFile(io.BytesIO(blob))
    names = zf.namelist()
    log(f"  zip members: {names[:8]}{'…' if len(names) > 8 else ''}")
    candidate = None
    for n in names:
        if n.rstrip("/").endswith("LoanData.csv") or n.lower().endswith("loandata.csv"):
            candidate = n
            break
    if candidate is None:
        # single csv member
        csvs = [n for n in names if n.lower().endswith(".csv")]
        if len(csvs) == 1:
            candidate = csvs[0]
    if candidate is None:
        raise RuntimeError(f"No LoanData.csv in Kaggle zip. Members={names}")

    dest_csv.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(candidate) as src, dest_csv.open("wb") as out:
        shutil.copyfileobj(src, out)
    return "downloaded_kaggle"


def ensure_loandata(dest: Path, timeout: int) -> dict:
    """Return status dict; raises if cannot obtain verified LoanData.csv."""
    expected_sha = cfg.BONDORA["loandata_sha256"]
    expected_bytes = int(cfg.BONDORA["loandata_bytes"])

    # 1) Already present + hash OK
    if dest.exists() and dest.stat().st_size > 1_000_000:
        digest = sha256_file(dest)
        if digest == expected_sha and dest.stat().st_size == expected_bytes:
            return {
                "status": "skipped_hash_ok",
                "path": str(dest.relative_to(cfg.ROOT)),
                "bytes": dest.stat().st_size,
                "sha256": digest,
                "source": "existing",
            }
        log(
            f"  WARNING: existing LoanData.csv hash/size mismatch "
            f"(sha={digest[:12]}… size={dest.stat().st_size}); will refresh"
        )

    # 2) Kaggle API
    status = download_kaggle_loandata(dest, timeout)
    digest = sha256_file(dest)
    ok = digest == expected_sha and dest.stat().st_size == expected_bytes
    if not ok:
        log(
            f"  WARNING: downloaded LoanData.csv sha/size differ from pin "
            f"(sha={digest} size={dest.stat().st_size}). "
            "Update cfg.BONDORA pin if this is an intentional newer dump."
        )
    return {
        "status": status,
        "path": str(dest.relative_to(cfg.ROOT)),
        "bytes": dest.stat().st_size,
        "sha256": digest,
        "source": cfg.DATA_URLS["bondora_loandata_kaggle"],
        "hash_matches_pin": ok,
    }


def write_source_md(dest: Path, loandata_info: dict) -> None:
    text = "\n".join(
        [
            "# Bondora LoanData — provenance (B01)",
            "",
            f"- **Canonical file:** `{dest.name}`",
            f"- **Kaggle API:** {cfg.DATA_URLS['bondora_loandata_kaggle']}",
            f"- **Dataset slug:** `{cfg.DATA_URLS['bondora_loandata_kaggle_slug']}`",
            f"- **SHA-256 (pinned):** `{cfg.BONDORA['loandata_sha256']}`",
            f"- **SHA-256 (local):** `{loandata_info.get('sha256')}`",
            f"- **Bytes:** {loandata_info.get('bytes')}",
            f"- **Acquire status:** {loandata_info.get('status')}",
            "",
            "Investor portal `loan_dataset_investor.xlsx` is optional/legacy (wrong schema for Cox).",
            "DOMAIN_02 uses this CSV via `D00_DOMAIN_02_align_loan_vintage.py`.",
            "",
        ]
    )
    (dest.parent / "SOURCE.md").write_text(text, encoding="utf-8")


def main() -> int:
    dest_dir = cfg.DIRS["raw_bondora"]
    dest_dir.mkdir(parents=True, exist_ok=True)
    log_dir = cfg.DIRS["logs"]
    log_dir.mkdir(parents=True, exist_ok=True)

    timeout = int(cfg.BONDORA["timeout_s"])
    chunk = int(cfg.BONDORA["chunk_size"])
    loan_csv = dest_dir / cfg.BONDORA["loandata_local_name"]
    xlsx_dest = dest_dir / cfg.BONDORA["loan_local_name"]

    log("─" * 60)
    log("B01 — BONDORA LOANDATA (DOMAIN_02 canonical)")
    log("─" * 60)
    log(f"  Dest     : {dest_dir.relative_to(cfg.ROOT)}")
    log(f"  Canonical: {loan_csv.name}")
    log(f"  Kaggle   : {cfg.DATA_URLS['bondora_loandata_kaggle']}")
    log(f"  Pin SHA  : {cfg.BONDORA['loandata_sha256'][:16]}…")
    log("─" * 60)

    log("\n[1/3] Canonical LoanData.csv")
    try:
        loandata_info = ensure_loandata(loan_csv, timeout)
    except Exception as exc:  # noqa: BLE001
        log(f"ERROR: could not obtain LoanData.csv — {type(exc).__name__}: {exc}")
        return 1
    log(
        f"  → {loandata_info['status']}  "
        f"({loandata_info['bytes'] / 1e6:.1f} MB)  "
        f"sha={loandata_info['sha256'][:12]}…"
    )
    write_source_md(loan_csv, loandata_info)

    log("\n[2/3] Optional investor xlsx (legacy schema; not used by DOMAIN_02 Cox)")
    xlsx_status = download_file(
        cfg.DATA_URLS["bondora_loan_xlsx"], xlsx_dest, timeout, chunk
    )
    log(f"  → {xlsx_status}")

    log("\n[3/3] Probe legacy Bondora zips / repayments")
    repay_probe = probe_url(cfg.DATA_URLS["bondora_repayments_zip_legacy"], timeout)
    loan_legacy_probe = probe_url(cfg.DATA_URLS["bondora_loan_zip_legacy"], timeout)
    log(
        f"  repayments legacy: available={repay_probe['available']}  "
        f"status={repay_probe.get('http_status')}"
    )
    log(
        f"  loan zip legacy  : available={loan_legacy_probe['available']}  "
        f"status={loan_legacy_probe.get('http_status')}"
    )

    hash_ok = loandata_info.get("sha256") == cfg.BONDORA["loandata_sha256"]
    status = {
        "stage": "B01",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "paper": {
            "baseline": cfg.DOMAINS["domain2"]["baseline"],
            "doi": cfg.DOMAINS["domain2"]["doi"],
            "datasets_cited": ["loan", "repayments"],
            "retrieved_by_authors": cfg.BONDORA["paper_retrieved"],
        },
        "loandata_canonical": {
            **loandata_info,
            "kaggle_url": cfg.DATA_URLS["bondora_loandata_kaggle"],
            "kaggle_slug": cfg.DATA_URLS["bondora_loandata_kaggle_slug"],
            "pinned_sha256": cfg.BONDORA["loandata_sha256"],
            "pinned_bytes": cfg.BONDORA["loandata_bytes"],
            "hash_matches_pin": hash_ok,
        },
        "loan_xlsx_optional": {
            "url": cfg.DATA_URLS["bondora_loan_xlsx"],
            "path": str(xlsx_dest.relative_to(cfg.ROOT)) if xlsx_dest.exists() else None,
            "bytes": xlsx_dest.stat().st_size if xlsx_dest.exists() else None,
            "download_status": xlsx_status,
            "note": "Wrong schema for paper Cox; not used by D00–D04",
        },
        "repayments": {
            "required_for": cfg.BONDORA["repayments_required_for"],
            "legacy_probe": repay_probe,
            "local_present": False,
            "note": "IRR skipped until RepaymentsData available",
        },
        "legacy_loan_zip_probe": loan_legacy_probe,
    }
    status_path = log_dir / "b01_bondora_status.json"
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    log(f"\n  Status JSON: {status_path.relative_to(cfg.ROOT)}")

    if not loan_csv.exists() or loan_csv.stat().st_size <= 0:
        log("ERROR: canonical LoanData.csv missing")
        return 1
    if not hash_ok:
        log(
            "ERROR: LoanData.csv SHA-256 does not match pin — "
            "refusing to claim notebook reproducibility of published DOMAIN_02 numbers."
        )
        return 1

    log("─" * 60)
    log("B01 complete — canonical LoanData.csv ready (hash verified).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
