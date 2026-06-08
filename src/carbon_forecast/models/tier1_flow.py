"""Tier 1 flow ANN: per-interconnector 96h net-flow forecast.

CarbonCast ignores cross-border exchange; modeling it explicitly is
Contribution 2 of the thesis. One feedforward model per interconnector
(partner zone), same shape as the source ANN (dense 50 -> 34 -> 24 -> 96h head,
ReLU, RMSE loss).

Target is SIGNED net flow with a partner: net = import - export (MW, positive =
net import). Because flow is legitimately bidirectional, predictions are NOT
clipped (unlike production, which is non-negative).

Inputs, via the windowing helper:
  - history (168h): the interconnector's own past net flow.
  - future-known (96h): calendar features plus this zone's GFS weather forecast.
    The locked spec also wants the partner endpoint's weather, but Phase 1 only
    stores single-centroid weather for the five modeled zones, so partner-side
    weather is omitted (documented deviation).

Normalization is per-interconnector and train-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import keras
import numpy as np
import pandas as pd

from carbon_forecast.data.normalize import Normalizer
from carbon_forecast.data.windowing import WindowedDataset, make_windows
from carbon_forecast.models.tier1_source import (
    WEATHER_FORECAST_COLS,
    build_source_ann,
)
from carbon_forecast.utils.calendar import CALENDAR_FEATURES


@dataclass
class Tier1FlowConfig:
    lookback: int = 168
    horizon: int = 96
    epochs: int = 100
    batch_size: int = 32
    stride: int = 1
    val_stride: int = 1
    patience: int = 10


@dataclass
class Tier1FlowArtifacts:
    model: keras.Model
    normalizer: Normalizer
    partner: str
    target_col: str
    history_cols: list[str]
    future_cols: list[str]
    config: Tier1FlowConfig
    history: dict = field(default_factory=dict)


def net_flow_col(partner: str) -> str:
    return f"net_flow_{partner}_mw"


def partners_for(frame: pd.DataFrame) -> list[str]:
    """Partner zones that appear in either import_ or export_ flow columns."""
    partners: set[str] = set()
    for c in frame.columns:
        for pre in ("import_", "export_"):
            if c.startswith(pre) and c.endswith("_mw") and c != f"{pre}total_mw":
                partners.add(c[len(pre):-len("_mw")])
    return sorted(partners)


def add_net_flows(frame: pd.DataFrame, partners: list[str] | None = None) -> pd.DataFrame:
    """Append signed net-flow columns (import - export, missing legs = 0)."""
    partners = partners or partners_for(frame)
    out = frame.copy()
    for p in partners:
        imp = out[f"import_{p}_mw"] if f"import_{p}_mw" in out else 0.0
        exp = out[f"export_{p}_mw"] if f"export_{p}_mw" in out else 0.0
        imp = imp.fillna(0.0) if isinstance(imp, pd.Series) else imp
        exp = exp.fillna(0.0) if isinstance(exp, pd.Series) else exp
        out[net_flow_col(p)] = imp - exp
    return out


def feature_plan_flow(partner: str, frame_cols: list[str]) -> tuple[str, list[str], list[str]]:
    """Return (target_col, history_cols, future_cols) for an interconnector."""
    target_col = net_flow_col(partner)
    history_cols = [target_col]
    future_cols = list(CALENDAR_FEATURES) + [c for c in WEATHER_FORECAST_COLS if c in frame_cols]
    return target_col, history_cols, future_cols


def _window(frame: pd.DataFrame, partner: str, cfg: Tier1FlowConfig, stride: int) -> WindowedDataset:
    target_col, history_cols, future_cols = feature_plan_flow(partner, list(frame.columns))
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


def train_flow_model(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    partner: str,
    cfg: Tier1FlowConfig | None = None,
    verbose: int = 1,
) -> Tier1FlowArtifacts:
    """Fit one interconnector ANN. Net-flow columns are derived first; the
    Normalizer is fit on train_frame only."""
    cfg = cfg or Tier1FlowConfig()
    train_frame = add_net_flows(train_frame, [partner])
    val_frame = add_net_flows(val_frame, [partner])
    target_col, history_cols, future_cols = feature_plan_flow(partner, list(train_frame.columns))

    normalizer = Normalizer.fit(train_frame)
    train_n = normalizer.transform(train_frame)
    val_n = normalizer.transform(val_frame)

    train_ds = _window(train_n, partner, cfg, cfg.stride)
    val_ds = _window(val_n, partner, cfg, cfg.val_stride)

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
    return Tier1FlowArtifacts(
        model=model, normalizer=normalizer, partner=partner, target_col=target_col,
        history_cols=history_cols, future_cols=future_cols, config=cfg,
        history=hist.history,
    )


def predict_flow_mw(
    artifacts: Tier1FlowArtifacts, frame: pd.DataFrame
) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Forecast signed net flow in MW (inverse-normalized). No clipping."""
    frame = add_net_flows(frame, [artifacts.partner])
    frame_n = artifacts.normalizer.transform(frame)
    ds = _window(frame_n, artifacts.partner, artifacts.config, stride=1)
    preds_norm = artifacts.model.predict(ds.flat_inputs(), verbose=0)
    preds_mw = artifacts.normalizer.inverse_transform(preds_norm, artifacts.target_col)
    return preds_mw, ds.origins


def evaluate_flow_mw(
    artifacts: Tier1FlowArtifacts, frame: pd.DataFrame
) -> dict[str, float]:
    """Flow accuracy in physical MW: MAE and RMSE (MAPE unsuitable; flow crosses 0)."""
    preds, _ = predict_flow_mw(artifacts, frame)
    frame = add_net_flows(frame, [artifacts.partner])
    frame_n = artifacts.normalizer.transform(frame)
    truth_ds = _window(frame_n, artifacts.partner, artifacts.config, stride=1)
    y_true = artifacts.normalizer.inverse_transform(truth_ds.y[..., 0], artifacts.target_col)
    err = preds - y_true
    return {
        "mae_mw": float(np.nanmean(np.abs(err))),
        "rmse_mw": float(np.sqrt(np.nanmean(err ** 2))),
        "n_samples": int(preds.shape[0]),
    }
