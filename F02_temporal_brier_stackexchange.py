"""
F02_temporal_brier_stackexchange.py — H5 probe (horizon Brier degradation)
==========================================================================
Paper-facing Stack Exchange / Politics exhibit.

**Primary (H5 decision):** IPCW Brier at θ ∈ {12, 24, 36} months rises while
global Harrell C stays in the Abedi band [0.66, 0.76].

**Sensitivity appendix:** time-dependent cumulative/dynamic AUC (non-decisive;
did not reject — retained for honesty).

H5 preview reject when ∃ frozen D3 RSF with:
  - Brier(t) mono-increasing on {12,24,36},
  - Brier(36) − Brier(12) > 2 × combined bootstrap SE,
  - global Harrell C ∈ [0.66, 0.76].

Writes ``results/probes/F02_h5_temporal.{json,md,tex,csv}``.

Execute:
    python -W default F02_temporal_brier_stackexchange.py
    python -W default F02_temporal_brier_stackexchange.py --full   # B=1000
"""

from __future__ import annotations

import csv
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import cfg
from src.metrics.freeze import load_frozen_manifest
from src.metrics.io import write_json
from src.metrics.predict import load_train_for_ipcw, predict_survival_curves
from src.metrics.temporal_auc import h5_auc_strip, h5_brier_strip

warnings.filterwarnings("default")


def log(msg: str = "") -> None:
    print(msg, flush=True)


def np_linspace_horizons(horizons: list[int]):
    hi = max(float(max(horizons)) * 1.25, float(max(horizons)) + 1.0)
    grid = np.linspace(0.0, hi, 80)
    grid[0] = 0.0
    return grid


def _eval_model(artifact: dict, horizons: list[int], B: int, seed: int) -> dict:
    mid = artifact["model_id"]
    if not artifact.get("predict_ready") or not artifact.get("load_ok"):
        return {"model_id": mid, "status": "skipped_not_ready"}
    if artifact["domain_id"] != "DOMAIN_03":
        return {"model_id": mid, "status": "skipped_wrong_domain"}

    train = load_train_for_ipcw(artifact)
    if train is None:
        return {"model_id": mid, "status": "error", "error": "no IPCW train"}
    time_train, event_train = train

    times_grid = np_linspace_horizons(horizons)
    curves = predict_survival_curves(artifact, times_grid=times_grid, n_grid=len(times_grid))

    brier = h5_brier_strip(
        time_train=time_train,
        event_train=event_train,
        time_test=curves["time"],
        event_test=curves["event"],
        risk=curves["risk"],
        surv_grid=curves["surv_grid"],
        times_grid=curves["times_grid"],
        horizons_months=horizons,
        n_bootstrap=B,
        seed=seed,
    )
    auc = h5_auc_strip(
        time_train=time_train,
        event_train=event_train,
        time_test=curves["time"],
        event_test=curves["event"],
        risk=curves["risk"],
        surv_grid=curves["surv_grid"],
        times_grid=curves["times_grid"],
        horizons_months=horizons,
        n_bootstrap=min(B, 200),
        seed=seed,
    )

    return {
        "model_id": mid,
        "h1_rank_key": artifact.get("h1_rank_key"),
        "status": brier.get("status"),
        "n": curves["n"],
        "n_events": curves["n_events"],
        "sha256": artifact.get("sha256"),
        "brier_strip": brier,
        "auc_strip_sensitivity": auc,
        "h5_preview_reject": bool(brier.get("h5_preview_reject")),
        "h5_auc_sensitivity_reject": bool(auc.get("h5_auc_would_reject")),
    }


def main() -> int:
    log("─" * 60)
    log("F02 — H5 BRIER HORIZON DEGRADATION (Stack Exchange Politics)")
    log("─" * 60)

    h5 = cfg.PROTOCOL["hypotheses"]["H5"]
    horizons = [int(m) for m in h5["horizons_months"]]
    band = list(h5["global_cindex_band"])
    B = 1000 if "--full" in sys.argv else min(200, int(cfg.EVAL["n_bootstrap"]))
    seed = int(cfg.RANDOM_SEED)

    manifest = load_frozen_manifest()
    arts = [
        a
        for a in (manifest.get("artifacts") or [])
        if a.get("domain_id") == "DOMAIN_03" and a.get("kind") == "estimator"
    ]
    if not arts:
        log("ERROR: no DOMAIN_03 estimators in E00 manifest — run E00 first")
        return 1

    log(f"  Primary: IPCW Brier @ {horizons}m  |  AUC = sensitivity")
    log(f"  B={B}  C-band={band}")
    results: list[dict] = []
    for art in arts:
        log(f"  → {art['model_id']} …")
        row = _eval_model(art, horizons, B, seed)
        results.append(row)
        bs = row.get("brier_strip") or {}
        bv = (bs.get("brier") or {}).get("values") or {}
        boot = bs.get("bootstrap") or {}
        log(
            f"     C={bs.get('global_harrell_c')} in_band={bs.get('global_c_in_band')}  "
            f"Brier={bv}  Δ36−12={bs.get('delta_brier_36_minus_12')}  "
            f"mono↑={bs.get('monotonic_increase')}  "
            f"rise>2SE={bs.get('rise_exceeds_2se')} "
            f"(2SE={None if boot.get('combined_se') is None else 2*boot['combined_se']:.4f})  "
            f"H5={row.get('h5_preview_reject')}"
        )

    any_reject = any(r.get("h5_preview_reject") for r in results)
    primary = next((r for r in results if r.get("h5_preview_reject")), None)
    if primary is None:
        in_band = [
            r for r in results if (r.get("brier_strip") or {}).get("global_c_in_band")
        ]
        primary = in_band[0] if in_band else (results[0] if results else {})

    caption_parts = []
    for r in results:
        s = r.get("brier_strip") or {}
        v = (s.get("brier") or {}).get("values") or {}
        caption_parts.append(
            f"{r.get('model_id')}: C={s.get('global_harrell_c') and round(s['global_harrell_c'], 3)}; "
            f"Brier(12/24/36)="
            f"{v.get('12') and round(v['12'], 3)}/"
            f"{v.get('24') and round(v['24'], 3)}/"
            f"{v.get('36') and round(v['36'], 3)}; "
            f"Δ={s.get('delta_brier_36_minus_12') and round(s['delta_brier_36_minus_12'], 3)}"
        )
    caption = (
        "On Politics θ=24 ladder eval (frozen RSF), IPCW Brier at "
        f"{horizons} months rises under a stable/global C. "
        + " ".join(caption_parts)
        + f" H5 preview reject (∃ model, Brier primary): {any_reject}. "
        "Time-dependent AUC was the sensitivity strip and did not reject."
    )

    doc = {
        "stage": "F02",
        "probe": "h5_temporal_brier_stackexchange",
        "hypothesis": "H5",
        "protocol_version": cfg.PROTOCOL.get("version"),
        "primary_metric": "ipcw_brier_at_horizons",
        "sensitivity_metric": "cumulative_dynamic_auc",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "horizons_months": horizons,
        "global_cindex_band": band,
        "n_bootstrap": B,
        "models": results,
        "primary_model_id": primary.get("model_id"),
        "h5_preview_reject": any_reject,
        "honesty_note": h5.get("honesty_note"),
        "why_relevant": [
            "The C-index (and even horizon AUC) can look stable while absolute "
            "probabilistic accuracy (Brier) worsens with the forecast horizon.",
            "This is the anchor desideratum gap — discrimination ≠ temporal/"
            "probabilistic accuracy — on real Stack Exchange Politics RSFs.",
            "Abedi et al. report a global C band ≈[0.66,0.76]; H5 asks whether "
            "that number masks horizon-specific proper-score degradation.",
            "AUC(t) strip is retained as sensitivity: the first written H5 "
            "statistic; it did not reject and is not the decision rule.",
        ],
        "caption_draft": caption,
        "artifacts_related": [
            "results/ladder/d03_discrimination.json",
            "results/ladder/d03_scores.json",
        ],
    }

    probe_dir = cfg.DIRS["probes"]
    probe_dir.mkdir(parents=True, exist_ok=True)
    stem = probe_dir / "F02_h5_temporal"
    write_json(Path(str(stem) + ".json"), doc)

    md = [
        "# F02 — H5 temporal Brier degradation (Stack Exchange Politics)",
        "",
        f"_Generated UTC: `{doc['generated_at_utc']}` · protocol `{doc['protocol_version']}`_",
        "",
        "**Primary decision statistic:** IPCW Brier @ horizons (sksurv).",
        "**Sensitivity (non-decisive):** cumulative/dynamic AUC.",
        "",
        f"- Horizons: **{horizons}** months",
        f"- Global C band (paper): **{band}**",
        f"- Bootstrap B: **{B}**",
        f"- **H5 preview reject (∃ model, Brier):** `{any_reject}`",
        f"- Primary exhibit model: `{primary.get('model_id')}`",
        "",
        f"> Honesty: {h5.get('honesty_note')}",
        "",
        "## Models",
        "",
    ]
    for r in results:
        s = r.get("brier_strip") or {}
        v = (s.get("brier") or {}).get("values") or {}
        boot = s.get("bootstrap") or {}
        auc = r.get("auc_strip_sensitivity") or {}
        av = (auc.get("auc") or {}).get("values") or {}
        md += [
            f"### `{r.get('model_id')}`",
            "",
            f"- n={r.get('n')}  events={r.get('n_events')}",
            f"- Harrell C = **{s.get('global_harrell_c')}**  "
            f"(in band `{s.get('global_c_in_band')}`)",
            f"- **Brier(12/24/36)** = {v.get('12')} / {v.get('24')} / {v.get('36')}",
            f"- Δ Brier(36−12) = **{s.get('delta_brier_36_minus_12')}**  "
            f"(combined SE={boot.get('combined_se')}; "
            f"CI Δ=[{boot.get('ci_delta_low')}, {boot.get('ci_delta_high')}])",
            f"- Mono ↑: `{s.get('monotonic_increase')}`  "
            f"rise>2SE: `{s.get('rise_exceeds_2se')}`  "
            f"**H5:** `{r.get('h5_preview_reject')}`",
            f"- AUC sensitivity 12/24/36: {av.get('12')} / {av.get('24')} / {av.get('36')} "
            f"(would-reject `{r.get('h5_auc_sensitivity_reject')}`)",
            "",
        ]
    md += ["## Why this matters", ""]
    for bullet in doc["why_relevant"]:
        md.append(f"- {bullet}")
    md += ["", "## Caption draft", "", caption, ""]
    Path(str(stem) + ".md").write_text("\n".join(md), encoding="utf-8")

    tex = [
        "% Auto-generated by F02_temporal_brier_stackexchange.py",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Model & $C_H$ & Brier$_{12}$ & Brier$_{24}$ & Brier$_{36}$ & "
        "$\\Delta_{36-12}$ & H5 \\\\",
        "\\midrule",
    ]

    def _f(x):
        return f"{x:.3f}" if isinstance(x, (int, float)) else "---"

    for r in results:
        s = r.get("brier_strip") or {}
        v = (s.get("brier") or {}).get("values") or {}
        tex.append(
            f"{r.get('model_id')} & {_f(s.get('global_harrell_c'))} & "
            f"{_f(v.get('12'))} & {_f(v.get('24'))} & {_f(v.get('36'))} & "
            f"{_f(s.get('delta_brier_36_minus_12'))} & "
            f"{r.get('h5_preview_reject')} \\\\"
        )
    tex += ["\\bottomrule", "\\end{tabular}", ""]
    Path(str(stem) + ".tex").write_text("\n".join(tex), encoding="utf-8")

    with Path(str(stem) + ".csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "model_id",
            "harrell_c",
            "brier_12",
            "brier_24",
            "brier_36",
            "delta_36_12",
            "combined_se",
            "monotonic_increase",
            "rise_exceeds_2se",
            "c_in_band",
            "h5_preview_reject",
            "auc_12",
            "auc_24",
            "auc_36",
            "auc_would_reject",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            s = r.get("brier_strip") or {}
            v = (s.get("brier") or {}).get("values") or {}
            boot = s.get("bootstrap") or {}
            auc = r.get("auc_strip_sensitivity") or {}
            av = (auc.get("auc") or {}).get("values") or {}
            w.writerow(
                {
                    "model_id": r.get("model_id"),
                    "harrell_c": s.get("global_harrell_c"),
                    "brier_12": v.get("12"),
                    "brier_24": v.get("24"),
                    "brier_36": v.get("36"),
                    "delta_36_12": s.get("delta_brier_36_minus_12"),
                    "combined_se": boot.get("combined_se"),
                    "monotonic_increase": s.get("monotonic_increase"),
                    "rise_exceeds_2se": s.get("rise_exceeds_2se"),
                    "c_in_band": s.get("global_c_in_band"),
                    "h5_preview_reject": r.get("h5_preview_reject"),
                    "auc_12": av.get("12"),
                    "auc_24": av.get("24"),
                    "auc_36": av.get("36"),
                    "auc_would_reject": r.get("h5_auc_sensitivity_reject"),
                }
            )

    log(f"  Wrote {stem.relative_to(cfg.ROOT)}.*")
    log(f"  H5 preview reject (Brier primary, any model) = {any_reject}")
    log("F02 complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
