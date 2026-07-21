"""
B00_download_backblaze.py — Download Backblaze Drive Stats zips
==============================================================
Downloads the zip archives needed for Domain 1 reproduction
(Ahmed & Green 2013–2022 window) into ``data/raw/backblaze/``.

Idempotent: skips files whose local size already matches the remote
``Content-Length``. Partial downloads resume via HTTP Range when supported.

Execute:
    python B00_download_backblaze.py

Done when:
    - Every zip listed in ``cfg.BACKBLAZE["zip_filenames"]`` exists locally
    - Local size equals remote Content-Length (when the server provides it)
    - Exit code 0 and no Python warnings under ``python -W default``
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests

from src.config import cfg

# Surface third-party issues during this stage (do not silence).
warnings.filterwarnings("default")


def log(msg: str = "") -> None:
    print(msg, flush=True)


def zip_url(filename: str) -> str:
    return cfg.DATA_URLS["backblaze_zip_template"].format(filename=filename)


def remote_size(url: str, timeout: int) -> int | None:
    """Return Content-Length if present, else None."""
    with requests.head(url, timeout=timeout, allow_redirects=True) as resp:
        resp.raise_for_status()
        length = resp.headers.get("Content-Length")
        if length is None:
            return None
        return int(length)


def download_one(filename: str, dest_dir: Path) -> str:
    """
    Download one zip. Returns status label:
    'skipped' | 'downloaded' | 'resumed'
    """
    url = zip_url(filename)
    dest = dest_dir / filename
    timeout = int(cfg.BACKBLAZE["timeout_s"])
    chunk = int(cfg.BACKBLAZE["chunk_size"])

    expected = remote_size(url, timeout)
    if dest.exists() and expected is not None and dest.stat().st_size == expected:
        return "skipped"
    if dest.exists() and expected is None and dest.stat().st_size > 0:
        # No remote length to verify; keep existing file.
        return "skipped"

    headers: dict[str, str] = {}
    mode = "wb"
    status = "downloaded"
    existing = dest.stat().st_size if dest.exists() else 0
    if existing > 0 and (expected is None or existing < expected):
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
        status = "resumed"

    with requests.get(
        url,
        stream=True,
        timeout=timeout,
        headers=headers,
        allow_redirects=True,
    ) as resp:
        # If server ignores Range, restart cleanly.
        if existing > 0 and resp.status_code == 200:
            mode = "wb"
            status = "downloaded"
            existing = 0
        resp.raise_for_status()

        total = expected
        if total is None:
            cl = resp.headers.get("Content-Length")
            if cl is not None:
                total = existing + int(cl) if status == "resumed" else int(cl)

        bytes_done = existing
        last_report = existing
        report_every = 50 << 20  # 50 MiB

        with dest.open(mode) as fh:
            for block in resp.iter_content(chunk_size=chunk):
                if not block:
                    continue
                fh.write(block)
                bytes_done += len(block)
                if bytes_done - last_report >= report_every:
                    if total:
                        pct = 100.0 * bytes_done / total
                        log(f"    … {filename}: {bytes_done / 1e6:.0f} MB / {total / 1e6:.0f} MB ({pct:.1f}%)")
                    else:
                        log(f"    … {filename}: {bytes_done / 1e6:.0f} MB")
                    last_report = bytes_done

    final_size = dest.stat().st_size
    if expected is not None and final_size != expected:
        raise RuntimeError(
            f"Size mismatch for {filename}: local={final_size} remote={expected}"
        )
    return status


def main() -> int:
    dest_dir = cfg.DIRS["raw_backblaze"]
    dest_dir.mkdir(parents=True, exist_ok=True)

    filenames = list(cfg.BACKBLAZE["zip_filenames"])
    log(f"{'─' * 60}")
    log("B00 — DOWNLOAD BACKBLAZE DRIVE STATS")
    log(f"{'─' * 60}")
    log(f"  Dest     : {dest_dir.relative_to(cfg.ROOT)}")
    log(f"  Period   : {cfg.BACKBLAZE['period_start']} → {cfg.BACKBLAZE['period_end']}")
    log(f"  Model    : {cfg.BACKBLAZE['model_filter']} (filter applied in later stages)")
    log(f"  Archives : {len(filenames)}")
    log(f"{'─' * 60}")

    counts = {"skipped": 0, "downloaded": 0, "resumed": 0}
    for i, name in enumerate(filenames, 1):
        log(f"\n[{i}/{len(filenames)}] {name}")
        status = download_one(name, dest_dir)
        counts[status] += 1
        size_mb = (dest_dir / name).stat().st_size / 1e6
        log(f"  → {status}  ({size_mb:.1f} MB)")

    missing = [n for n in filenames if not (dest_dir / n).exists()]
    if missing:
        log("\nERROR: missing files after download:")
        for n in missing:
            log(f"  - {n}")
        return 1

    log(f"\n{'─' * 60}")
    log(
        f"Summary: downloaded={counts['downloaded']}  "
        f"resumed={counts['resumed']}  skipped={counts['skipped']}"
    )
    log("B00 complete — Backblaze raw zips ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
