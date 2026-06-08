"""Windowing: processed hourly frame -> supervised (input, target) tensors.

Turns a contiguous hourly frame into sliding windows. Each sample is anchored
at a forecast-origin time t:

  - history window  rows [t - lookback, t)   -> past covariates (e.g. 168h of
    source production, historical CI)
  - future window   rows [t, t + horizon)    -> known-ahead covariates (96h
    weather forecast, calendar)
  - target window   rows [t, t + horizon)    -> the values to predict (96h)

A dense Tier-1 ANN flattens history+future to a flat vector; the Tier-2
CNN-LSTM consumes the 3-D history sequence directly. This helper stays
model-agnostic: it returns raw aligned arrays and the origin timestamps.

NaN handling is the caller's job (e.g. fill no-flow interconnectors with 0
before windowing); `drop_na=True` additionally skips any window that still
contains NaN so partial-coverage edges never reach the model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class WindowedDataset:
    """Aligned supervised windows.

    Shapes:
      X_hist  (n_samples, lookback, n_history_features)
      X_fut   (n_samples, horizon,  n_future_features)  or None
      y       (n_samples, horizon,  n_targets)
      origins (n_samples,)  forecast-origin timestamp t of each sample
    """

    X_hist: np.ndarray
    X_fut: np.ndarray | None
    y: np.ndarray
    origins: pd.DatetimeIndex
    history_cols: list[str]
    future_cols: list[str]
    target_cols: list[str]
    lookback: int
    horizon: int

    @property
    def n_samples(self) -> int:
        return self.X_hist.shape[0]

    def flat_inputs(self) -> np.ndarray:
        """Flatten history (+future) into one vector per sample, for dense ANNs."""
        parts = [self.X_hist.reshape(self.n_samples, -1)]
        if self.X_fut is not None:
            parts.append(self.X_fut.reshape(self.n_samples, -1))
        return np.concatenate(parts, axis=1)


def _check_hourly_contiguous(index: pd.DatetimeIndex) -> None:
    if not index.is_monotonic_increasing:
        raise ValueError("frame index must be sorted ascending.")
    diffs = index[1:] - index[:-1]
    if len(diffs) and (diffs != pd.Timedelta(hours=1)).any():
        raise ValueError(
            "frame index has gaps or non-hourly steps; reindex to a complete "
            "hourly range before windowing so windows never span a gap."
        )


def _windows(arr: np.ndarray, size: int) -> np.ndarray:
    """Sliding windows over axis 0: (n-size+1, size, n_features)."""
    sw = np.lib.stride_tricks.sliding_window_view(arr, size, axis=0)
    return np.moveaxis(sw, -1, 1)


def make_windows(
    frame: pd.DataFrame,
    *,
    history_cols: list[str],
    target_cols: list[str],
    future_cols: list[str] | None = None,
    lookback: int = 168,
    horizon: int = 96,
    stride: int = 1,
    drop_na: bool = True,
    dtype: type = np.float32,
) -> WindowedDataset:
    """Build sliding-window tensors from a contiguous hourly frame."""
    _check_hourly_contiguous(frame.index)
    n = len(frame)
    if n < lookback + horizon:
        raise ValueError(
            f"need at least lookback+horizon={lookback + horizon} rows, got {n}."
        )
    future_cols = future_cols or []

    hist = frame[history_cols].to_numpy(dtype=dtype)
    tgt = frame[target_cols].to_numpy(dtype=dtype)
    fut = frame[future_cols].to_numpy(dtype=dtype) if future_cols else None

    # Valid origin t in [lookback, n - horizon]; both window stacks share length.
    n_valid = n - horizon - lookback + 1
    X_hist = _windows(hist, lookback)[:n_valid]            # origins t-lookback
    y = _windows(tgt, horizon)[lookback:lookback + n_valid]  # origins t
    X_fut = (
        _windows(fut, horizon)[lookback:lookback + n_valid] if fut is not None else None
    )
    origins = frame.index[lookback:lookback + n_valid]

    if stride > 1:
        sel = slice(None, None, stride)
        X_hist, y, origins = X_hist[sel], y[sel], origins[sel]
        if X_fut is not None:
            X_fut = X_fut[sel]

    if drop_na:
        ok = ~np.isnan(X_hist).any(axis=(1, 2)) & ~np.isnan(y).any(axis=(1, 2))
        if X_fut is not None:
            ok &= ~np.isnan(X_fut).any(axis=(1, 2))
        X_hist, y, origins = X_hist[ok], y[ok], origins[ok]
        if X_fut is not None:
            X_fut = X_fut[ok]

    return WindowedDataset(
        X_hist=X_hist,
        X_fut=X_fut,
        y=y,
        origins=origins,
        history_cols=list(history_cols),
        future_cols=list(future_cols),
        target_cols=list(target_cols),
        lookback=lookback,
        horizon=horizon,
    )
