"""
Rung 1 — Discrimination (Harrell / Antolini / Uno + C-index Multiverse strip).

Harrell: sksurv ``concordance_index_censored`` (risk: higher = worse).
Uno: sksurv ``concordance_index_ipcw`` — same path validated on anchor Figura 5.
Antolini: stub until survival-curve path is wired in E.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from sksurv.util import Surv

from src.config import cfg

VARIANTS = list(cfg.EVAL["cindex_variants"])


def _structured(event: np.ndarray, time: np.ndarray):
    return Surv.from_arrays(event.astype(bool), time.astype(float))


def cindex_harrell(
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    **_kwargs: Any,
) -> dict[str, Any]:
    c, _, _, _, _ = concordance_index_censored(
        event.astype(bool), time.astype(float), risk.astype(float)
    )
    return {"metric": "harrell_c", "status": "live", "value": float(c)}


def cindex_uno(
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    train_time: np.ndarray | None = None,
    train_event: np.ndarray | None = None,
    tau: float | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    if train_time is None or train_event is None:
        train_time, train_event = time, event
    y_train = _structured(train_event, train_time)
    y_test = _structured(event, time)
    if tau is None:
        tau = float(np.quantile(train_time, 0.8))
    try:
        c, _, _, _, _ = concordance_index_ipcw(
            y_train, y_test, risk.astype(float), tau=tau
        )
        return {
            "metric": "uno_c",
            "status": "live",
            "value": float(c),
            "tau": tau,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "metric": "uno_c",
            "status": "error",
            "value": None,
            "tau": tau,
            "error": str(exc),
        }


def cindex_antolini(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    return {
        "metric": "antolini_c",
        "status": "stub",
        "value": None,
        "note": "Needs time-dependent survival curves; wire with E02/E03 predict path.",
    }


def multiverse_strip(
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
    train_time: np.ndarray | None = None,
    train_event: np.ndarray | None = None,
) -> dict[str, Any]:
    """Harrell vs Uno sensitivity (Antolini pending)."""
    h = cindex_harrell(time, event, risk)
    u = cindex_uno(time, event, risk, train_time=train_time, train_event=train_event)
    a = cindex_antolini()
    values = {
        "harrell": h.get("value"),
        "uno": u.get("value"),
        "antolini": a.get("value"),
    }
    return {
        "metric": "cindex_multiverse",
        "status": "live" if h.get("value") is not None else "stub",
        "variants": VARIANTS,
        "values": values,
        "details": {"harrell": h, "uno": u, "antolini": a},
    }
