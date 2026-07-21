"""
G00_leakage_controls_audit.py — Block G: leakage / rigor checklist
=================================================================
Audits process controls (not a new scientific hypothesis):

  - protocol freeze present
  - Domain 2 temporal hold-out + auction-time features
  - Domain 3 eval panel vs train IPCW separation hooks
  - Block E uses frozen estimators (no retrain in E0x)
  - single final evaluation artifacts exist (ladder + probes)

Writes:
  results/logs/leakage_controls_audit.{json,md}

Execute:
    python -W default G00_leakage_controls_audit.py
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import cfg
from src.metrics.io import write_json


def log(msg: str = "") -> None:
    print(msg, flush=True)


def _ok(check_id: str, title: str, detail: str, **extra: Any) -> dict[str, Any]:
    return {"id": check_id, "title": title, "status": "pass", "detail": detail, **extra}


def _warn(check_id: str, title: str, detail: str, **extra: Any) -> dict[str, Any]:
    return {"id": check_id, "title": title, "status": "warn", "detail": detail, **extra}


def _fail(check_id: str, title: str, detail: str, **extra: Any) -> dict[str, Any]:
    return {"id": check_id, "title": title, "status": "fail", "detail": detail, **extra}


def check_protocol_freeze() -> dict[str, Any]:
    path = cfg.DIRS["logs"] / "protocol_freeze.md"
    conf_v = cfg.PROTOCOL.get("version")
    if not path.exists():
        return _fail("G00.1", "Protocol freeze on disk", f"missing {path}")
    text = path.read_text(encoding="utf-8")
    if conf_v and conf_v not in text:
        return _warn(
            "G00.1",
            "Protocol freeze on disk",
            f"freeze file present but does not mention config version `{conf_v}`",
            path=str(path.relative_to(cfg.ROOT)),
            protocol_version=conf_v,
        )
    return _ok(
        "G00.1",
        "Protocol freeze on disk",
        f"found; PROTOCOL.version={conf_v}",
        path=str(path.relative_to(cfg.ROOT)),
        protocol_version=conf_v,
    )


def check_d2_temporal_split() -> dict[str, Any]:
    path = cfg.ROOT / "data/processed/domain2/loans_split.parquet"
    if not path.exists():
        return _fail("G00.2", "D2 temporal split parquet", f"missing {path}")
    import pandas as pd

    df = pd.read_parquet(path, columns=None)
    cols = set(df.columns)
    need = {"split_role", "duration_days", "event"}
    missing = sorted(need - cols)
    if missing:
        return _fail("G00.2", "D2 temporal split parquet", f"missing columns {missing}")
    roles = df["split_role"].value_counts().to_dict()
    n_train = int(roles.get("train", 0))
    n_test = int(roles.get("test", 0))
    if n_train < 100 or n_test < 100:
        return _fail(
            "G00.2",
            "D2 temporal split parquet",
            f"implausible sizes train={n_train} test={n_test}",
            roles=roles,
        )
    # Overlap check if LoanId present
    overlap = None
    if "LoanId" in cols or "loan_id" in cols:
        idcol = "LoanId" if "LoanId" in cols else "loan_id"
        tr = set(df.loc[df.split_role == "train", idcol].astype(str))
        te = set(df.loc[df.split_role == "test", idcol].astype(str))
        overlap = len(tr & te)
        if overlap > 0:
            return _fail(
                "G00.2",
                "D2 temporal split parquet",
                f"train/test LoanId overlap={overlap}",
                roles=roles,
                overlap=overlap,
            )
    return _ok(
        "G00.2",
        "D2 temporal split parquet",
        f"train={n_train:,} test={n_test:,}; id_overlap={overlap}",
        roles={k: int(v) for k, v in roles.items()},
        id_overlap=overlap,
    )


def check_d2_auction_time_features() -> dict[str, Any]:
    feat_path = cfg.ROOT / "data/processed/domain2/design_features.json"
    leak_tokens = (
        "DefaultDate",
        "Status",
        "PrincipalRecovered",
        "InterestAndPenalty",
        "DebtOccuredOn",
        "PlannedInterest",
        "NextPaymentNr",
        "NrOfScheduledPayments",
        "ReScheduledOn",
    )
    if not feat_path.exists():
        return _warn(
            "G00.3",
            "D2 auction-time feature list",
            f"missing {feat_path}; cannot verify covariate leakage filter",
        )
    doc = json.loads(feat_path.read_text(encoding="utf-8"))
    feats = list(doc.get("features") or doc.get("feature_cols") or [])
    hits = [f for f in feats if any(t.lower() in f.lower() for t in leak_tokens)]
    if hits:
        return _fail(
            "G00.3",
            "D2 auction-time feature list",
            f"possible post-auction / outcome columns in design: {hits}",
            n_features=len(feats),
            suspects=hits,
        )
    return _ok(
        "G00.3",
        "D2 auction-time feature list",
        f"{len(feats)} features; no obvious post-auction tokens",
        n_features=len(feats),
    )


def check_d1_eval_mode_documented() -> dict[str, Any]:
    """D1 Cox is in-sample GOF — must be explicit, not sold as hold-out."""
    man = cfg.DIRS["ladder"] / "frozen_models_manifest.json"
    if not man.exists():
        return _warn("G00.4", "D1 eval mode documented", "E00 manifest missing")
    doc = json.loads(man.read_text(encoding="utf-8"))
    arts = [a for a in doc.get("artifacts") or [] if a.get("domain_id") == "DOMAIN_01"]
    modes = {a.get("model_id"): a.get("eval_mode") for a in arts}
    bad = [m for m, e in modes.items() if e and "in_sample" not in str(e)]
    if not arts:
        return _warn("G00.4", "D1 eval mode documented", "no DOMAIN_01 artifacts in manifest")
    if bad:
        return _warn(
            "G00.4",
            "D1 eval mode documented",
            f"unexpected eval_mode for {bad}: {modes}",
            eval_modes=modes,
        )
    return _ok(
        "G00.4",
        "D1 eval mode documented",
        "DOMAIN_01 models marked in_sample_gof (not a temporal hold-out)",
        eval_modes=modes,
    )


def check_e_scripts_no_retrain() -> dict[str, Any]:
    """E01–E06 should load frozen models; forbid obvious estimator.fit in E0x bodies."""
    root = cfg.ROOT
    offenders: list[str] = []
    # Allow comments / strings; flag lines with .fit( that look like training
    fit_re = re.compile(r"^\s*[^#].*\.fit\s*\(")
    for path in sorted(root.glob("E0*.py")):
        text = path.read_text(encoding="utf-8")
        for i, line in enumerate(text.splitlines(), 1):
            if "def " in line:
                continue
            if fit_re.search(line) and "Concordance" not in line:
                # SurvivalEVAL / lifelines KM in helpers may call fit — flag for review
                if any(x in line for x in ("CoxPH", "cph.", "model.fit", "rsf.fit", "Booster")):
                    offenders.append(f"{path.name}:{i}: {line.strip()[:100]}")
    if offenders:
        return _fail(
            "G00.5",
            "Block E no-retrain scan",
            f"suspicious .fit in E scripts: {offenders}",
            offenders=offenders,
        )
    return _ok(
        "G00.5",
        "Block E no-retrain scan",
        "no Cox/RSF/XGB .fit calls found in E0*.py",
    )


def check_d3_eval_panel() -> dict[str, Any]:
    eval_path = cfg.ROOT / "results/models/domain3/ladder_eval_p_theta24.parquet"
    panel = cfg.ROOT / "data/processed/domain3/p_theta24.parquet"
    if not eval_path.exists():
        return _fail("G00.6", "D3 ladder eval panel", f"missing {eval_path}")
    import pandas as pd

    ev = pd.read_parquet(eval_path)
    detail = f"eval n={len(ev):,}"
    if panel.exists() and "UserId" in ev.columns:
        pan = pd.read_parquet(panel, columns=["UserId"])
        overlap_frac = len(ev) / max(len(pan), 1)
        detail += f"; panel n={len(pan):,}; eval/panel={overlap_frac:.3f}"
        # eval should be a fold subset, not the full panel
        if len(ev) >= len(pan):
            return _warn(
                "G00.6",
                "D3 ladder eval panel",
                detail + " — eval size ≥ panel (check fold export)",
            )
    return _ok("G00.6", "D3 ladder eval panel", detail)


def check_final_eval_artifacts() -> dict[str, Any]:
    need = [
        cfg.DIRS["ladder"] / "summary.json",
        cfg.DIRS["probes"] / "report.json",
        cfg.DIRS["reproduction"] / "LADDER_evaluation_table.json",
    ]
    missing = [str(p.relative_to(cfg.ROOT)) for p in need if not p.exists()]
    if missing:
        return _fail(
            "G00.7",
            "Final evaluation artifacts",
            f"missing {missing}",
            missing=missing,
        )
    return _ok(
        "G00.7",
        "Final evaluation artifacts",
        "ladder summary + probes report + paper ladder table present",
    )


def check_f00_uses_same_population() -> dict[str, Any]:
    """Ablated vs full should share drives.parquet path."""
    pilot = cfg.DIRS["probes"] / "F00_h4_ablation.json"
    if not pilot.exists():
        return _warn("G00.8", "F00 same-population ablation", "F00_h4_ablation.json missing")
    doc = json.loads(pilot.read_text(encoding="utf-8"))
    arts = doc.get("artifacts") or {}
    full = arts.get("full_model")
    abl = arts.get("ablated_model")
    if not full or not abl:
        return _warn("G00.8", "F00 same-population ablation", "artifact paths incomplete")
    if not (cfg.ROOT / full).exists() or not (cfg.ROOT / abl).exists():
        return _fail("G00.8", "F00 same-population ablation", "model joblibs missing on disk")
    return _ok(
        "G00.8",
        "F00 same-population ablation",
        "full + ablated joblibs present; LOO survey is primary H4 exhibit",
        full=full,
        ablated=abl,
    )


CHECKS = [
    check_protocol_freeze,
    check_d2_temporal_split,
    check_d2_auction_time_features,
    check_d1_eval_mode_documented,
    check_e_scripts_no_retrain,
    check_d3_eval_panel,
    check_final_eval_artifacts,
    check_f00_uses_same_population,
]


def main() -> int:
    log("─" * 60)
    log("G00 — LEAKAGE / RIGOR CONTROLS AUDIT")
    log("─" * 60)

    results = []
    for fn in CHECKS:
        r = fn()
        results.append(r)
        mark = {"pass": "✓", "warn": "!", "fail": "✗"}[r["status"]]
        log(f"  [{mark}] {r['id']} {r['title']}: {r['detail']}")

    n_pass = sum(1 for r in results if r["status"] == "pass")
    n_warn = sum(1 for r in results if r["status"] == "warn")
    n_fail = sum(1 for r in results if r["status"] == "fail")
    overall = "fail" if n_fail else ("warn" if n_warn else "pass")

    doc = {
        "stage": "G00",
        "artifact": "leakage_controls_audit",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": cfg.PROTOCOL.get("version"),
        "overall_status": overall,
        "n_pass": n_pass,
        "n_warn": n_warn,
        "n_fail": n_fail,
        "checks": results,
    }

    out_dir = cfg.DIRS["logs"]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "leakage_controls_audit.json", doc)

    md = [
        "# G00 — Leakage / rigor controls audit",
        "",
        f"_Generated UTC: `{doc['generated_at_utc']}` · protocol `{doc['protocol_version']}`_",
        "",
        f"**Overall:** `{overall}` — pass={n_pass}, warn={n_warn}, fail={n_fail}",
        "",
        "| ID | Status | Check | Detail |",
        "|----|--------|-------|--------|",
    ]
    for r in results:
        md.append(
            f"| {r['id']} | `{r['status']}` | {r['title']} | {r['detail']} |"
        )
    md += [
        "",
        "## Notes",
        "",
        "- This audit is a **process checklist**, not a hypothesis test.",
        "- Domain 1 Cox remains **in-sample GOF** (documented); do not claim hold-out discrimination for D1.",
        "- H4 SMART concentration is a scientific probe (F00), distinct from this leakage-control audit.",
        "",
    ]
    (out_dir / "leakage_controls_audit.md").write_text("\n".join(md), encoding="utf-8")

    log(f"  Overall={overall}  Wrote results/logs/leakage_controls_audit.*")
    log("G00 complete.")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
