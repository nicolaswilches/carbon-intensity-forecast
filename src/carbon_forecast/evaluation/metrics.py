"""Forecast accuracy metrics for carbon intensity.

Pure, array-in / float-out helpers shared across the project. They match the
definitions used inside the model orchestrators: MAPE is the mean absolute
percentage error in percent (carbon intensity is bounded away from zero, so it is
well behaved), MAE and RMSE are in the target units (gCO2eq/kWh). NaNs are ignored
so partial windows do not void a score.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-6


def _err(preds, y_true) -> np.ndarray:
    return np.asarray(preds, dtype=float) - np.asarray(y_true, dtype=float)


def mae(preds, y_true) -> float:
    """Mean absolute error."""
    return float(np.nanmean(np.abs(_err(preds, y_true))))


def rmse(preds, y_true) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.nanmean(_err(preds, y_true) ** 2)))


def mape_pct(preds, y_true, eps: float = EPS) -> float:
    """Mean absolute percentage error, in percent."""
    denom = np.clip(np.abs(np.asarray(y_true, dtype=float)), eps, None)
    return float(np.nanmean(np.abs(_err(preds, y_true)) / denom) * 100)


def summary(preds, y_true) -> dict[str, float]:
    """All three metrics plus the sample count, as one dict."""
    return {
        "mape_pct": mape_pct(preds, y_true),
        "mae": mae(preds, y_true),
        "rmse": rmse(preds, y_true),
        "n_samples": int(np.asarray(preds).shape[0]),
    }


def per_horizon(preds, y_true, metric: str = "mape", eps: float = EPS) -> np.ndarray:
    """Per-horizon metric over (N, H) arrays, returning a length-H vector."""
    p = np.asarray(preds, dtype=float)
    y = np.asarray(y_true, dtype=float)
    if metric == "mape":
        return np.nanmean(np.abs(p - y) / np.clip(np.abs(y), eps, None), axis=0) * 100
    if metric == "mae":
        return np.nanmean(np.abs(p - y), axis=0)
    if metric == "rmse":
        return np.sqrt(np.nanmean((p - y) ** 2, axis=0))
    raise ValueError(f"unknown metric {metric!r}; use mape, mae, or rmse")
