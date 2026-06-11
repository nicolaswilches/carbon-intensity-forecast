"""Tier 2 CNN-LSTM: 96h carbon-intensity forecast.

Combines Tier 1 forecasts with historical CI, weather, and calendar into one
carbon-intensity forecast. Architecture (CarbonCast Fig. 5):
Conv1D(4) -> MaxPool -> Conv1D(16) -> LSTM(24) -> Dropout(0.1) -> Dense(96).
RMSE loss, Adam. Direct 96h output head (locked: not CarbonCast's iterated 24x4).

Input representation. A single multivariate sequence of length lookback+horizon
(168 + 96 = 264). Each timestep carries the same channels:
  - ci:           the CI target, ACTUAL in the past block, 0 in the future block
                  (it is what we predict).
  - dynamic:      channels that are ACTUAL in the past block and a Tier 1 / borrowed
                  FORECAST in the future block. For E2 these are the per-source
                  productions; E3 also adds interconnector net-flow forecasts and
                  partner-zone CI. The orchestrator injects the future values; when
                  it does not, the future block falls back to the frame's own actual
                  future values (perfect-forecast placeholder).
  - weather:      actual (past) / forecast (future), known across the window.
  - calendar:     known across the whole window.
  - is_future:    0 for the 168 past steps, 1 for the 96 future steps.
The LSTM thus sees the observed-to-forecast transition; the dense head emits
all 96 hours at once.

Target and dynamic channels are config-driven. E2 (CarbonCast-faithful) targets
production-based CI with source-only dynamics; E3 retargets to consumption-based
CI and widens the dynamic set to include flows and partner CI.
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

# E2 default target (CarbonCast-faithful, production-based CI).
E2_TARGET = "prod_based_ci_lifecycle"
# E3 target (consumption-based CI, EM's reported series).
E3_TARGET = "cons_based_ci"


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
    # CI series the Tier 2 head predicts.
    target_col: str = E2_TARGET
    # Forecastable channels (actual past / forecast future). None auto-detects
    # the per-source production columns, i.e. the E2 set.
    dynamic_cols: list[str] | None = None


@dataclass
class Tier2Artifacts:
    model: keras.Model
    normalizer: Normalizer
    target_col: str
    dynamic_cols: list[str]
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


def default_dynamic_cols(frame_cols: list[str]) -> list[str]:
    """E2 dynamic set: the per-source production columns."""
    return [c for c in frame_cols if c.startswith("prod_") and c.endswith("_mw")]


def _feature_sets(
    frame_cols: list[str], cfg: Tier2Config
) -> tuple[list[str], list[str], list[str]]:
    dynamic_cols = cfg.dynamic_cols if cfg.dynamic_cols is not None else default_dynamic_cols(frame_cols)
    weather_cols = [c for c in WEATHER_FORECAST_COLS if c in frame_cols]
    calendar_cols = list(CALENDAR_FEATURES)
    return dynamic_cols, weather_cols, calendar_cols


def assemble_sequences(
    frame_norm: pd.DataFrame,
    cfg: Tier2Config,
    stride: int,
    future_dynamic: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex, list[str]]:
    """Build the (N, 264, F) sequence tensor and (N, 96) target from a
    normalized frame.

    future_dynamic: optional (N, horizon, n_dynamic) array of Tier 1 / borrowed
    forecasts for the future block, ordered like the resolved dynamic columns.
    When None, the future block uses the frame's actual future values
    (perfect-forecast placeholder).
    """
    target_col = cfg.target_col
    dynamic_cols, weather_cols, calendar_cols = _feature_sets(list(frame_norm.columns), cfg)
    hist_cols = [target_col] + dynamic_cols + weather_cols + calendar_cols
    fut_cols = dynamic_cols + weather_cols + calendar_cols

    ds = make_windows(
        frame_norm,
        history_cols=hist_cols,
        target_cols=[target_col],
        future_cols=fut_cols,
        lookback=cfg.lookback,
        horizon=cfg.horizon,
        stride=stride,
        drop_na=True,
    )
    n = ds.n_samples

    # Past block channels already ordered [ci, dynamic, weather, calendar].
    past = ds.X_hist
    past = np.concatenate([past, np.zeros((n, cfg.lookback, 1), np.float32)], axis=2)  # is_future=0

    # Future block: prepend ci=0, optionally override the leading dynamic
    # channels, append is_future=1. Only the first K = future_dynamic.shape[-1]
    # channels are overridden, so dynamic groups with no model forecast (e.g.
    # E3's partner CI under the strategy-A placeholder) must be ordered last and
    # keep the frame's actual future values.
    fut_core = ds.X_fut.copy()  # [dynamic, weather, calendar]
    if future_dynamic is not None:
        k = future_dynamic.shape[-1]
        fut_core[:, :, :k] = future_dynamic.astype(np.float32)
    ci_zero = np.zeros((n, cfg.horizon, 1), np.float32)
    is_fut = np.ones((n, cfg.horizon, 1), np.float32)
    future = np.concatenate([ci_zero, fut_core, is_fut], axis=2)

    X = np.concatenate([past, future], axis=1)  # (N, 264, F)
    y = ds.y[..., 0]
    channels = ["ci"] + dynamic_cols + weather_cols + calendar_cols + ["is_future"]
    return X, y, ds.origins, channels


def train_tier2(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    cfg: Tier2Config | None = None,
    normalizer: Normalizer | None = None,
    train_future_dynamic: np.ndarray | None = None,
    val_future_dynamic: np.ndarray | None = None,
    verbose: int = 1,
) -> Tier2Artifacts:
    """Fit the Tier 2 CNN-LSTM. Normalizer fit on train_frame if not provided.

    train_future_dynamic / val_future_dynamic: optional (N, horizon, n_dynamic)
    forecasts for the future block. The orchestrator passes out-of-fold / final
    Tier 1 forecasts (and, for E3, borrowed partner forecasts); when None, the
    perfect-forecast placeholder (actual future values) is used.
    """
    cfg = cfg or Tier2Config()
    normalizer = normalizer or Normalizer.fit(train_frame)
    train_n = normalizer.transform(train_frame)
    val_n = normalizer.transform(val_frame)

    Xtr, ytr, _, channels = assemble_sequences(train_n, cfg, cfg.stride, train_future_dynamic)
    Xva, yva, _, _ = assemble_sequences(val_n, cfg, cfg.val_stride, val_future_dynamic)

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
    dynamic_cols, weather_cols, calendar_cols = _feature_sets(list(train_frame.columns), cfg)
    return Tier2Artifacts(
        model=model, normalizer=normalizer, target_col=cfg.target_col,
        dynamic_cols=dynamic_cols, weather_cols=weather_cols, calendar_cols=calendar_cols,
        config=cfg, history=hist.history,
    )


def predict_ci(
    artifacts: Tier2Artifacts, frame: pd.DataFrame, future_dynamic: np.ndarray | None = None
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Forecast 96h carbon intensity in gCO2eq/kWh (inverse-normalized, clipped >=0)."""
    frame_n = artifacts.normalizer.transform(frame)
    X, _, origins, _ = assemble_sequences(frame_n, artifacts.config, 1, future_dynamic)
    preds_norm = artifacts.model.predict(X, verbose=0)
    preds = artifacts.normalizer.inverse_transform(preds_norm, artifacts.target_col)
    return np.clip(preds, 0.0, None), origins


def evaluate_ci(
    artifacts: Tier2Artifacts, frame: pd.DataFrame, future_dynamic: np.ndarray | None = None
) -> dict[str, float]:
    """CI accuracy: MAPE (primary), MAE, RMSE in gCO2eq/kWh. MAPE is valid here
    because carbon intensity is bounded away from zero."""
    preds, _ = predict_ci(artifacts, frame, future_dynamic)
    frame_n = artifacts.normalizer.transform(frame)
    _, y_norm, _, _ = assemble_sequences(frame_n, artifacts.config, 1, future_dynamic)
    y_true = artifacts.normalizer.inverse_transform(y_norm, artifacts.target_col)
    err = preds - y_true
    denom = np.clip(np.abs(y_true), 1e-6, None)
    return {
        "mape_pct": float(np.nanmean(np.abs(err) / denom) * 100),
        "mae": float(np.nanmean(np.abs(err))),
        "rmse": float(np.sqrt(np.nanmean(err ** 2))),
        "n_samples": int(preds.shape[0]),
    }
