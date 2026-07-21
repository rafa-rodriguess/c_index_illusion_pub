"""
Frozen-model inventory for Block E (E00).

E scripts must load models via the manifest written by ``E00_load_frozen_models``.
Re-running E00 after a Domain lane refresh updates hashes — downstream E steps
should be re-executed afterward.

Ladder rung scripts (anchor double-helix parallel):
  E01_rung1_discrimination … E05_rung5_competing_risks
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from src.config import cfg

FROZEN_MANIFEST_NAME = "frozen_models_manifest.json"
PROTOCOL_FREEZE_NAME = "protocol_freeze.json"

# Registry of Block-D artifacts the ladder expects.
# ``required_for_e00`` False = optional until F00 / model export lands.
EXPECTED_ARTIFACTS: list[dict[str, Any]] = [
    {
        "domain_id": "DOMAIN_01",
        "model_id": "cox_full",
        "h1_rank_key": "cox_full",
        "kind": "estimator",
        "backend": "lifelines.CoxPHFitter",
        "path": "results/models/domain1/cox_ahmed_green.joblib",
        "metrics_path": "results/models/domain1/cox_metrics.json",
        "eval_data_path": "data/processed/domain1/drives_cox_cohort.parquet",
        "time_col": "duration_days",
        "event_col": "event",
        "eval_mode": "in_sample_gof",
        "required_for_e00": True,
        "notes": "Ahmed & Green Cox PH (in-sample GOF in Fase A).",
    },
    {
        "domain_id": "DOMAIN_01",
        "model_id": "cox_ablated",
        "h1_rank_key": "cox_ablated",
        "kind": "estimator",
        "backend": "lifelines.CoxPHFitter",
        "path": "results/models/domain1/cox_ablated_h4.joblib",
        "metrics_path": "results/models/domain1/cox_ablated_h4_metrics.json",
        "eval_data_path": "data/processed/domain1/drives_cox_cohort.parquet",
        "time_col": "duration_days",
        "event_col": "event",
        "eval_mode": "in_sample_gof",
        "required_for_e00": False,
        "notes": "Created by F00 / H4 ablation — not required to start E skeleton.",
    },
    {
        "domain_id": "DOMAIN_02",
        "model_id": "cox_classical",
        "h1_rank_key": "cox_classical",
        "kind": "estimator",
        "backend": "lifelines.CoxPHFitter",
        "path": "results/models/domain2/cox_linear.joblib",
        "metrics_path": "results/models/domain2/metrics.json",
        "eval_data_path": "results/models/domain2/test_scored.parquet",
        "features_path": "data/processed/domain2/design_features.json",
        "time_col": "duration_days",
        "event_col": "event",
        "eval_mode": "temporal_test_2020",
        "required_for_e00": True,
        "notes": "Bone-Winkel linear Cox.",
    },
    {
        "domain_id": "DOMAIN_02",
        "model_id": "cox_xgboost",
        "h1_rank_key": "cox_xgboost",
        "kind": "estimator",
        "backend": "xgboost.XGBRegressor_cox",
        "path": "results/models/domain2/xgb_cox.joblib",
        "metrics_path": "results/models/domain2/metrics.json",
        "eval_data_path": "results/models/domain2/test_scored.parquet",
        "features_path": "data/processed/domain2/design_features.json",
        "time_col": "duration_days",
        "event_col": "event",
        "eval_mode": "temporal_test_2020",
        "required_for_e00": True,
        "notes": "Bone-Winkel XGB-Cox + Breslow S(t) via D03b (no Booster retrain).",
    },
    {
        "domain_id": "DOMAIN_03",
        "model_id": "rsf_behavioural",
        "h1_rank_key": "rsf_behavioural",
        "kind": "estimator",
        "backend": "sksurv.ensemble.RandomSurvivalForest",
        "path": "results/models/domain3/rsf_behavioural_p_theta24.joblib",
        "metrics_path": "results/models/domain3/ladder_rsf_export.json",
        "eval_data_path": "results/models/domain3/ladder_eval_p_theta24.parquet",
        "time_col": "duration_months",
        "event_col": "event",
        "eval_mode": "cv_fold0_run0_p_theta24",
        "required_for_e00": True,
        "notes": "Ladder freeze: Politics θ=24 behavioural RSF (D02b export from current panels).",
    },
    {
        "domain_id": "DOMAIN_03",
        "model_id": "rsf_content",
        "h1_rank_key": "rsf_content",
        "kind": "estimator",
        "backend": "sksurv.ensemble.RandomSurvivalForest",
        "path": "results/models/domain3/rsf_content_p_theta24.joblib",
        "metrics_path": "results/models/domain3/ladder_rsf_export.json",
        "eval_data_path": "results/models/domain3/ladder_eval_p_theta24.parquet",
        "time_col": "duration_months",
        "event_col": "event",
        "eval_mode": "cv_fold0_run0_p_theta24",
        "required_for_e00": True,
        "notes": "Ladder freeze: Politics θ=24 content RSF (D02b).",
    },
    {
        "domain_id": "DOMAIN_03",
        "model_id": "rsf_combined",
        "h1_rank_key": "rsf_combined",
        "kind": "estimator",
        "backend": "sksurv.ensemble.RandomSurvivalForest",
        "path": "results/models/domain3/rsf_combined_p_theta24.joblib",
        "metrics_path": "results/models/domain3/ladder_rsf_export.json",
        "eval_data_path": "results/models/domain3/ladder_eval_p_theta24.parquet",
        "time_col": "duration_months",
        "event_col": "event",
        "eval_mode": "cv_fold0_run0_p_theta24",
        "required_for_e00": True,
        "notes": "Ladder freeze: Politics θ=24 combined RSF (D02b).",
    },
    {
        "domain_id": "DOMAIN_03",
        "model_id": "rsf_fold_scores",
        "h1_rank_key": None,
        "kind": "cv_scores",
        "backend": "sksurv.RSF_cv_scores",
        "path": "results/models/domain3/rsf_fold_scores.csv",
        "metrics_path": "results/models/domain3/rsf_metrics.json",
        "eval_data_path": "data/processed/domain3/p_theta24.parquet",
        "time_col": None,
        "event_col": None,
        "eval_mode": "cv_scores_only",
        "required_for_e00": False,
        "notes": "Table-8 CV score dump (provenance); ladder prediction uses D02b joblibs.",
    },
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def protocol_freeze_path() -> Path:
    return cfg.DIRS["logs"] / PROTOCOL_FREEZE_NAME


def frozen_manifest_path() -> Path:
    return cfg.DIRS["ladder"] / FROZEN_MANIFEST_NAME


def require_protocol_freeze() -> dict[str, Any]:
    path = protocol_freeze_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.relative_to(cfg.ROOT)} — run C00_preregister_protocol.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_frozen_manifest() -> dict[str, Any]:
    path = frozen_manifest_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.relative_to(cfg.ROOT)} — run E00_load_frozen_models.py first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _feature_list(root: Path, spec: dict[str, Any]) -> list[str] | None:
    fp = spec.get("features_path")
    if not fp:
        return None
    path = root / fp
    if not path.exists():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    feats = doc.get("features")
    return list(feats) if feats else None


def sanity_load_estimator(root: Path, abs_path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    """Load joblib blob; return load_ok + feature metadata. Does not predict."""
    import joblib

    try:
        blob = joblib.load(abs_path)
    except Exception as exc:  # noqa: BLE001
        return {"load_ok": False, "load_error": str(exc), "n_features": None, "feature_cols": None}

    feature_cols: list[str] | None = None
    model_type = type(blob).__name__

    if isinstance(blob, dict):
        model_type = type(blob.get("model", blob)).__name__
        if "smart_cols" in blob:
            feature_cols = list(blob["smart_cols"])
        elif "features" in blob:
            feature_cols = list(blob["features"])
    else:
        # bare xgb / estimator
        feature_cols = _feature_list(root, spec)

    if feature_cols is None:
        feature_cols = _feature_list(root, spec)

    return {
        "load_ok": True,
        "load_error": None,
        "blob_type": model_type,
        "n_features": len(feature_cols) if feature_cols else None,
        "feature_cols": feature_cols,
    }


def inventory_artifacts(root: Path | None = None, *, sanity_load: bool = True) -> list[dict[str, Any]]:
    root = root or cfg.ROOT
    rows: list[dict[str, Any]] = []
    for spec in EXPECTED_ARTIFACTS:
        rel = Path(spec["path"])
        abs_path = root / rel
        metrics_rel = Path(spec["metrics_path"]) if spec.get("metrics_path") else None
        metrics_abs = (root / metrics_rel) if metrics_rel else None
        eval_rel = Path(spec["eval_data_path"]) if spec.get("eval_data_path") else None
        eval_abs = (root / eval_rel) if eval_rel else None
        present = abs_path.exists()
        entry: dict[str, Any] = {
            **spec,
            "present": present,
            "sha256": sha256_file(abs_path) if present else None,
            "bytes": abs_path.stat().st_size if present else None,
            "metrics_present": bool(metrics_abs and metrics_abs.exists()),
            "eval_data_present": bool(eval_abs and eval_abs.exists()),
            "predict_ready": present and spec["kind"] == "estimator",
            "load_ok": None,
            "load_error": None,
            "blob_type": None,
            "n_features": None,
            "feature_cols": None,
        }
        if present and spec["kind"] == "estimator" and sanity_load:
            loaded = sanity_load_estimator(root, abs_path, spec)
            entry.update(loaded)
            if not loaded["load_ok"]:
                entry["predict_ready"] = False
        elif present and spec["kind"] == "cv_scores":
            entry["load_ok"] = True
            entry["blob_type"] = "cv_scores_csv"
        rows.append(entry)
    return rows
