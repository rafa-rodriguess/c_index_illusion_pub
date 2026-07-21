"""
AN00_ANCHOR_fetch_code.py — Fetch Lillelund et al. position-cindex code
=======================================================================
Harness-check lane (not Domain 04). Downloads the author repo artefacts
into ``data/raw/anchor/position-cindex/`` so we can re-run their synthetic
Weibull + Clayton experiment and match Table 2 / Figure 5.

Role: calibrate our Block E metric implementations against the anchor's
oracle-accessible DGP. Does **not** replace D1–D3 reproduction.

Execute:
    python -W default AN00_ANCHOR_fetch_code.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parent))

import requests

from src.config import cfg

warnings.filterwarnings("default")

RAW_BASE = "https://raw.githubusercontent.com/{repo}/{ref}/{path}"


def log(msg: str = "") -> None:
    print(msg, flush=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_file(rel_path: str, dest: Path, *, timeout: int = 120) -> dict:
    url = RAW_BASE.format(
        repo=cfg.ANCHOR["code_repo"],
        ref=cfg.ANCHOR["code_ref"],
        path=quote(rel_path, safe="/"),
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        data = dest.read_bytes()
        return {
            "path": rel_path,
            "status": "cached",
            "bytes": len(data),
            "sha256": sha256_bytes(data),
            "url": url,
        }
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return {
        "path": rel_path,
        "status": "downloaded",
        "bytes": len(r.content),
        "sha256": sha256_bytes(r.content),
        "url": url,
    }


def main() -> int:
    log("─" * 60)
    log("AN00 — FETCH ANCHOR AUTHOR CODE (harness check)")
    log("─" * 60)

    out_dir: Path = cfg.ANCHOR["raw_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    lit = cfg.ANCHOR["literature_dir"]
    lit.mkdir(parents=True, exist_ok=True)

    results = []
    for rel in cfg.ANCHOR["author_files"]:
        dest = out_dir / rel
        try:
            info = download_file(rel, dest)
            log(f"  [{info['status']:<10}] {rel}  ({info['bytes']} B)")
            results.append(info)
        except Exception as exc:  # noqa: BLE001 — surface per-file failure
            log(f"  [FAILED    ] {rel}: {exc}")
            results.append({"path": rel, "status": "failed", "error": str(exc)})

    failed = [r for r in results if r.get("status") == "failed"]
    manifest = {
        "stage": "AN00",
        "role": "anchor_harness_check",
        "repo": cfg.ANCHOR["code"],
        "ref": cfg.ANCHOR["code_ref"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "out_dir": str(out_dir.relative_to(cfg.ROOT)),
        "n_ok": len(results) - len(failed),
        "n_failed": len(failed),
        "files": results,
    }
    man_path = cfg.DIRS["logs"] / "an00_anchor_fetch.json"
    man_path.parent.mkdir(parents=True, exist_ok=True)
    man_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    access_md = lit / "CODE_ACCESS.md"
    access_md.write_text(
        "\n".join(
            [
                "# Anchor code access — Lillelund et al. (ICML 2026 Spotlight)",
                "",
                f"- Repo: {cfg.ANCHOR['code']}",
                f"- Local mirror: `{out_dir.relative_to(cfg.ROOT)}/` (via AN00)",
                f"- Role: **harness check** for Block E metrics (synthetic Weibull + Clayton).",
                "- Not a fourth domain baseline.",
                "",
                f"Last fetch UTC: `{manifest['generated_at_utc']}` — "
                f"{manifest['n_ok']} ok / {manifest['n_failed']} failed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    log(f"  Wrote {man_path.relative_to(cfg.ROOT)}")
    log(f"  Wrote {access_md.relative_to(cfg.ROOT)}")

    if failed:
        log(f"AN00 incomplete — {len(failed)} file(s) failed.")
        return 1
    log("AN00 complete — author code mirrored.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
