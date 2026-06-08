"""Tier 1 source ANN: per-source 96h production forecast (CarbonCast-faithful).

One feedforward model per generation source. Architecture (CarbonCast 3.4):
dense 50 -> 34 -> 24 with ReLU, then a 96-unit dense head producing the full
96h forecast in a single pass (direct multi-output, NOT CarbonCast's iterative
24h-by-4 scheme; locked decision to avoid error compounding).

Inputs, assembled via the windowing helper:
  - history (168h): the source's own past production.
  - future-known (96h): calendar features, plus the GFS weather forecast for
    RENEWABLE sources only (solar/wind/hydro), per CarbonCast.
The dense net consumes these flattened into one vector.

Normalization is per-source and train-only: the frame is standardized with a
Normalizer fit on the train fold before windowing; predictions are
inverse-transformed back to MW for reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import keras
import numpy as np
import pandas as pd

from carbon_forecast.data.normalize import Normalizer
from carbon_forecast.data.windowing import WindowedDataset, make_windows
from carbon_forecast.utils.calendar import CALENDAR_FEATURES

# Sources that receive weather covariates (CarbonCast: renewables only).
RENEWABLE_SOURCES = {"wind", "solar", "hydro"}

# GFS forecast variables fed to renewable models (CarbonCast 3.3 set).
WEATHER_FORECAST_COLS = [
    "temperature_2m", "dewpoint_2m", "shortwave_radiation",
    "precipitation", "wind_u_10m", "wind_v_10m",
]

DENSE_UNITS = (50, 34, 24)


@dataclass
class Tier1Config:
    lookback: int = 168
    horizon: int = 96
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    stride: int = 1
    val_stride: int = 1
    patience: int = 10  # early-stopping patience on val loss


@dataclass
class Tier1Artifacts:
    model: keras.Model
    normalizer: Normalizer
    source: str
    target_col: str
    history_cols: list[str]
    future_cols: list[str]
    config: Tier1Config
    history: dict = field(default_factory=dict)


def _rmse(y_true, y_pred):
    return keras.ops.sqrt(keras.ops.mean(keras.ops.square(y_true - y_pred)))


def build_source_ann(input_dim: int, horizon: int = 96) -> keras.Model:
    """Dense 50->34->24->horizon, ReLU, RMSE loss, Adam."""
    model = keras.Sequential(
        [keras.layers.Input(shape=(input_dim,))]
        + [keras.layers.Dense(u, activation="relu") for u in DENSE_UNITS]
        + [keras.layers.Dense(horizon)],
        name="tier1_source_ann",
    )
    model.compile(optimizer=keras.optimizers.Adam(), loss=_rmse)
    return model


def feature_plan(source: str, frame_cols: list[str]) -> tuple[str, list[str], list[str]]:
    """Return (target_col, history_cols, future_cols) for a source."""
    target_col = f"prod_{source}_mw"
    history_cols = [target_col]
    future_cols = list(CALENDAR_FEATURES)
    if source in RENEWABLE_SOURCES:
        future_cols += [c for c in WEATHER_FORECAST_COLS if c in frame_cols]
    return target_col, history_cols, future_cols


def _window(frame: pd.DataFrame, source: str, cfg: Tier1Config, stride: int) -> WindowedDataset:
    target_col, history_cols, future_cols = feature_plan(source, list(frame.columns))
    return make_windows(
        frame,
        history_cols=history_cols,
        target_cols=[target_col],
        future_cols=future_cols,
        lookback=cfg.lookback,
        horizon=cfg.horizon,
        stride=stride,
        drop_na=True,
    )


def train_source_model(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    source: str,
    cfg: Tier1Config | None = None,
    verbose: int = 1,
) -> Tier1Artifacts:
    """Fit one source ANN. Normalizer is fit on train_frame only."""
    cfg = cfg or Tier1Config()
    target_col, history_cols, future_cols = feature_plan(source, list(train_frame.columns))

    # Train-only standardization, applied to both folds before windowing.
    normalizer = Normalizer.fit(train_frame)
    train_n = normalizer.transform(train_frame)
    val_n = normalizer.transform(val_frame)

    train_ds = _window(train_n, source, cfg, cfg.stride)
    val_ds = _window(val_n, source, cfg, cfg.val_stride)

    Xtr, ytr = train_ds.flat_inputs(), train_ds.y[..., 0]
    Xva, yva = val_ds.flat_inputs(), val_ds.y[..., 0]

    model = build_source_ann(Xtr.shape[1], cfg.horizon)
    es = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=cfg.patience, restore_best_weights=True
    )
    hist = model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=[es],
        verbose=verbose,
    )
    return Tier1Artifacts(
        model=model, normalizer=normalizer, source=source, target_col=target_col,
        history_cols=history_cols, future_cols=future_cols, config=cfg,
        history=hist.history,
    )


def predict_mw(artifacts: Tier1Artifacts, frame: pd.DataFrame) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Forecast in physical MW (inverse-normalized). Returns (preds, origins).

    Predictions are clipped to >= 0: production is non-negative, but the linear
    output head (CarbonCast-faithful) can emit small negatives. The clip is
    post-processing, not a model change.
    """
    frame_n = artifacts.normalizer.transform(frame)
    ds = _window(frame_n, artifacts.source, artifacts.config, stride=1)
    preds_norm = artifacts.model.predict(ds.flat_inputs(), verbose=0)
    preds_mw = artifacts.normalizer.inverse_transform(preds_norm, artifacts.target_col)
    return np.clip(preds_mw, 0.0, None), ds.origins


def evaluate_mw(
    artifacts: Tier1Artifacts, frame: pd.DataFrame
) -> dict[str, float]:
    """Tier-1 accuracy in physical MW: MAE and RMSE.

    MAPE is deliberately NOT reported here: intermittent sources dip to ~0 MW,
    where MAPE explodes and misleads. MAPE is reserved for the Tier-2 carbon
    intensity output, which is bounded away from zero (CarbonCast convention).
    """
    preds, origins = predict_mw(artifacts, frame)
    frame_n = artifacts.normalizer.transform(frame)
    truth_ds = _window(frame_n, artifacts.source, artifacts.config, stride=1)
    y_true = artifacts.normalizer.inverse_transform(truth_ds.y[..., 0], artifacts.target_col)
    err = preds - y_true
    return {
        "mae_mw": float(np.nanmean(np.abs(err))),
        "rmse_mw": float(np.sqrt(np.nanmean(err ** 2))),
        "n_samples": int(preds.shape[0]),
    }
