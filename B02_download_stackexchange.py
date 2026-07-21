"""
B02_download_stackexchange.py — Download Abedi Stack Exchange artefacts
=======================================================================
Domain 3 (Abedi Firouzjaei, 2022) releases preprocessed experiment data and
raw community extracts on GitHub. The roadmap prioritises these artefacts for
faithful RSF reproduction (C-index 0.66–0.76).

Downloads into ``data/raw/stackexchange/`` mirroring the repo layout:
  processed_data/{p,ds,cs}/user_features.csv
  raw_data/{p,ds,cs}/{users,questions,answers,comments}.csv.gz

Archive.org full dumps are optional upstream and are not fetched here.

Execute:
    python B02_download_stackexchange.py

Done when:
    - Every path in ``cfg.STACKEXCHANGE["files"]`` exists locally with size > 0
    - Exit code 0 under ``python -W default``
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests

from src.config import cfg

warnings.filterwarnings("default")


def log(msg: str = "") -> None:
    print(msg, flush=True)


def file_url(rel_path: str) -> str:
    return cfg.STACKEXCHANGE["url_template"].format(path=rel_path)


def remote_size(url: str, timeout: int) -> int | None:
    with requests.head(url, timeout=timeout, allow_redirects=True) as resp:
        resp.raise_for_status()
        length = resp.headers.get("Content-Length")
        return int(length) if length is not None else None


def download_one(rel_path: str, dest_root: Path) -> str:
    """Return 'skipped' | 'downloaded' | 'resumed'."""
    url = file_url(rel_path)
    dest = dest_root / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)

    timeout = int(cfg.STACKEXCHANGE["timeout_s"])
    chunk = int(cfg.STACKEXCHANGE["chunk_size"])

    expected = remote_size(url, timeout)
    if dest.exists() and expected is not None and dest.stat().st_size == expected:
        return "skipped"
    if dest.exists() and expected is None and dest.stat().st_size > 0:
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
        report_every = 5 << 20  # 5 MiB

        with dest.open(mode) as fh:
            for block in resp.iter_content(chunk_size=chunk):
                if not block:
                    continue
                fh.write(block)
                bytes_done += len(block)
                if total and bytes_done - last_report >= report_every:
                    pct = 100.0 * bytes_done / total
                    log(
                        f"    … {rel_path}: "
                        f"{bytes_done / 1e6:.1f} MB / {total / 1e6:.1f} MB ({pct:.1f}%)"
                    )
                    last_report = bytes_done

    if dest.stat().st_size <= 0:
        raise RuntimeError(f"Empty download: {rel_path}")
    if expected is not None and dest.stat().st_size != expected:
        raise RuntimeError(
            f"Size mismatch for {rel_path}: "
            f"local={dest.stat().st_size} remote={expected}"
        )
    return status


def main() -> int:
    dest_root = cfg.DIRS["raw_stackexchange"]
    dest_root.mkdir(parents=True, exist_ok=True)
    files = list(cfg.STACKEXCHANGE["files"])

    log(f"{'─' * 60}")
    log("B02 — DOWNLOAD STACK EXCHANGE / ABEDI ARTEFACTS")
    log(f"{'─' * 60}")
    log(f"  Dest     : {dest_root.relative_to(cfg.ROOT)}")
    log(f"  Source   : {cfg.STACKEXCHANGE['github_repo']}")
    log(f"  Period   : inception → {cfg.STACKEXCHANGE['period_end']}")
    log(f"  Files    : {len(files)}")
    log(f"{'─' * 60}")

    counts = {"skipped": 0, "downloaded": 0, "resumed": 0}
    for i, rel in enumerate(files, 1):
        log(f"\n[{i}/{len(files)}] {rel}")
        status = download_one(rel, dest_root)
        counts[status] += 1
        size = (dest_root / rel).stat().st_size
        log(f"  → {status}  ({size / 1e6:.2f} MB)")

    missing = [r for r in files if not (dest_root / r).exists() or (dest_root / r).stat().st_size <= 0]
    if missing:
        log("\nERROR: missing or empty files:")
        for r in missing:
            log(f"  - {r}")
        return 1

    # Sanity: three processed feature tables present
    for comm in cfg.STACKEXCHANGE["communities"]:
        feat = dest_root / "processed_data" / comm / "user_features.csv"
        if not feat.exists():
            log(f"\nERROR: missing processed features for community '{comm}'")
            return 1

    log(f"\n{'─' * 60}")
    log(
        f"Summary: downloaded={counts['downloaded']}  "
        f"resumed={counts['resumed']}  skipped={counts['skipped']}"
    )
    log("B02 complete — Stack Exchange / Abedi raw artefacts ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
