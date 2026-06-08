"""Tier 2 CNN-LSTM: 96h carbon-intensity forecast (CarbonCast-faithful, E2).

Combines Tier 1 source production forecasts with historical CI, weather, and
calendar into one carbon-intensity forecast. Architecture (CarbonCast Fig. 5):
Conv1D(4) -> MaxPool -> Conv1D(16) -> LSTM(24) -> Dropout(0.1) -> Dense(96).
RMSE loss, Adam. Direct 96h output head (locked: not CarbonCast's iterated 24x4).

Input representation. A single multivariate sequence of length lookback+horizon
(168 + 96 = 264). Each timestep carries the same channels:
  - ci:           production-based CI, ACTUAL in the past block, 0 in the future
                  block (it is what we predict).
  - prod_<src>:   ACTUAL production in the past block, Tier 1 FORECAST in the
                  future block (perfect-forecast placeholder until the
                  orchestrator injects real Tier 1 outputs).
  - weather:      actual (past) / forecast (future).
  - calendar:     known across the whole window.
  - is_future:    0 for the 168 past steps, 1 for the 96 future steps.
The LSTM thus sees the observed-to-forecast transition; the dense head emits
all 96 hours at once.

For E2 the target is production-based CI; E3 will retarget to consumption-based
CI and add flow-forecast channels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import keras
import numpy as np
import pandas as pd

from carbon_forecast.data.normalize import Normalizer
from carbon_forecast.data.windowing import make_windows
from carbon_forecast.models.tier1_source import WEATHER_FORECAST_COLS
from carbon_forecast.utils.calendar import CALENDAR_FEATURES

E2_TARGET = "prod_based_ci_lifecycle"


@dataclass
class Tier2Config:
    lookback: int = 168
    horizon: int = 96
    epochs: int = 100
    batch_size: int = 32
    dropout: float = 0.1
    stride: int = 1
    val_stride: int = 1
    patience: int = 10


@dataclass
class Tier2Artifacts:
    model: keras.Model
    normalizer: Normalizer
    target_col: str
    source_cols: list[str]
    weather_cols: list[str]
    calendar_cols: list[str]
    config: Tier2Config
    history: dict = field(default_factory=dict)


def _rmse(y_true, y_pred):
    return keras.ops.sqrt(keras.ops.mean(keras.ops.square(y_true - y_pred)))


def build_cnn_lstm(timesteps: int, n_features: int, cfg: Tier2Config) -> keras.Model:
    """CarbonCast Fig. 5 stack producing a 96h CI vector."""
    model = keras.Sequential(
        [
            keras.layers.Input(shape=(timesteps, n_features)),
            keras.layers.Conv1D(4, 4, padding="same", activation="relu"),
            keras.layers.MaxPooling1D(2, padding="same"),
            keras.layers.Conv1D(16, 4, padding="same", activation="relu"),
            keras.layers.LSTM(24),
            keras.layers.Dropout(cfg.dropout),
            keras.layers.Dense(cfg.horizon),
        ],
        name="tier2_cnn_lstm",
    )
    model.compile(optimizer=keras.optimizers.Adam(), loss=_rmse)
    return model


def _feature_sets(frame_cols: list[str]) -> tuple[list[str], list[str], list[str]]:
    source_cols = [c for c in frame_cols if c.startswith("prod_") and c.endswith("_mw")]
    weather_cols = [c for c in WEATHER_FORECAST_COLS if c in frame_cols]
    calendar_cols = list(CALENDAR_FEATURES)
    return source_cols, weather_cols, calendar_cols


def assemble_sequences(
    frame_norm: pd.DataFrame,
    cfg: Tier2Config,
    stride: int,
    future_source: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, list[str]]:
    """Build the (N, 264, F) sequence tensor and (N, 96) target from a
    normalized frame.

    future_source: optional (N, horizon, n_sources) array of Tier 1 production
    forecasts for the future block. When None, the future block uses the
    frame's actual future production (perfect-forecast placeholder).
    """
    source_cols, weather_cols, calendar_cols = _feature_sets(list(frame_norm.columns))
    hist_cols = [E2_TARGET] + source_cols + weather_cols + calendar_cols
    fut_cols = source_cols + weather_cols + calendar_cols

    ds = make_windows(
        frame_norm,
        history_cols=hist_cols,
        target_cols=[E2_TARGET],
        future_cols=fut_cols,
        lookback=cfg.lookback,
        horizon=cfg.horizon,
        stride=stride,
        drop_na=True,
    )
    n, S, W = ds.n_samples, len(source_cols), len(weather_cols)

    # Past block channels already ordered [ci, sources, weather, calendar].
    past = ds.X_hist
    past = np.concatenate([past, np.zeros((n, cfg.lookback, 1), np.float32)], axis=2)  # is_future=0

    # Future block: prepend ci=0, optionally override source channels, append is_future=1.
    fut_core = ds.X_fut.copy()  # [sources, weather, calendar]
    if future_source is not None:
        fut_core[:, :, :S] = future_source.astype(np.float32)
    ci_zero = np.zeros((n, cfg.horizon, 1), np.float32)
    is_fut = np.ones((n, cfg.horizon, 1), np.float32)
    future = np.concatenate([ci_zero, fut_core, is_fut], axis=2)

    X = np.concatenate([past, future], axis=1)  # (N, 264, F)
    y = ds.y[..., 0]
    channels = ["ci"] + source_cols + weather_cols + calendar_cols + ["is_future"]
    return X, y, ds.origins, channels


def train_tier2(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    cfg: Tier2Config | None = None,
    normalizer: Normalizer | None = None,
    verbose: int = 1,
) -> Tier2Artifacts:
    """Fit the Tier 2 CNN-LSTM. Normalizer fit on train_frame if not provided."""
    cfg = cfg or Tier2Config()
    normalizer = normalizer or Normalizer.fit(train_frame)
    train_n = normalizer.transform(train_frame)
    val_n = normalizer.transform(val_frame)

    Xtr, ytr, _, channels = assemble_sequences(train_n, cfg, cfg.stride)
    Xva, yva, _, _ = assemble_sequences(val_n, cfg, cfg.val_stride)

    model = build_cnn_lstm(Xtr.shape[1], Xtr.shape[2], cfg)
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
    source_cols, weather_cols, calendar_cols = _feature_sets(list(train_frame.columns))
    return Tier2Artifacts(
        model=model, normalizer=normalizer, target_col=E2_TARGET,
        source_cols=source_cols, weather_cols=weather_cols, calendar_cols=calendar_cols,
        config=cfg, history=hist.history,
    )


def predict_ci(
    artifacts: Tier2Artifacts, frame: pd.DataFrame, future_source: np.ndarray | None = None
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Forecast 96h carbon intensity in gCO2eq/kWh (inverse-normalized, clipped >=0)."""
    frame_n = artifacts.normalizer.transform(frame)
    X, _, origins, _ = assemble_sequences(frame_n, artifacts.config, 1, future_source)
    preds_norm = artifacts.model.predict(X, verbose=0)
    preds = artifacts.normalizer.inverse_transform(preds_norm, artifacts.target_col)
    return np.clip(preds, 0.0, None), origins


def evaluate_ci(
    artifacts: Tier2Artifacts, frame: pd.DataFrame, future_source: np.ndarray | None = None
) -> dict[str, float]:
    """CI accuracy: MAPE (primary), MAE, RMSE in gCO2eq/kWh. MAPE is valid here
    because carbon intensity is bounded away from zero."""
    preds, _ = predict_ci(artifacts, frame, future_source)
    frame_n = artifacts.normalizer.transform(frame)
    _, y_norm, _, _ = assemble_sequences(frame_n, artifacts.config, 1, future_source)
    y_true = artifacts.normalizer.inverse_transform(y_norm, artifacts.target_col)
    err = preds - y_true
    denom = np.clip(np.abs(y_true), 1e-6, None)
    return {
        "mape_pct": float(np.nanmean(np.abs(err) / denom) * 100),
        "mae": float(np.nanmean(np.abs(err))),
        "rmse": float(np.sqrt(np.nanmean(err ** 2))),
        "n_samples": int(preds.shape[0]),
    }
