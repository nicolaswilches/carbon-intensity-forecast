"""E3 orchestrator: CarbonCast-extended consumption-based CI model.

Extends the E2 baseline along the three thesis contributions:

  - Target: consumption-based CI (cons_based_ci), not production-based. This is
    the operationally relevant signal and the one Electricity Maps publishes.
  - Cross-border flow (Contribution 2): a Tier 1 flow ANN per interconnector
    forecasts signed net flow. These join the source ANNs as Tier 2 future-block
    channels, so the CNN-LSTM sees expected imports/exports, not just domestic
    generation.
  - Partner carbon intensity: each partner zone's CI feeds Tier 2 as a
    consumption-based channel (imported electricity carries the exporter's CI).

Same two-job structure as E2 (see carboncast_faithful): out-of-fold Tier 1
forecasts train Tier 2 honestly; final Tier 1 models drive inference. The flow
ANNs follow the source ANNs through both jobs.

Partner CI has no Tier 1 model (we borrow EM's published partner forecast at
inference), so it never enters the out-of-fold machinery. Strategy A: during
training the partner's own actual future CI is the placeholder (a perfect-forecast
stand-in, mirroring the source convention); at inference the future partner CI is
a persistence forecast (last observed value held flat) unless a real forecast is
supplied. Dynamic channels are ordered [sources, flows, partner_ci] so the
forecast-overridden channels lead and partner CI can fall back to actuals.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from carbon_forecast.data.normalize import Normalizer
from carbon_forecast.data.windowing import make_windows
from carbon_forecast.models import tier2_cnnlstm as t2
from carbon_forecast.models.tier1_flow import (
    Tier1FlowArtifacts,
    Tier1FlowConfig,
    predict_flow_mw,
    train_flow_model,
)
from carbon_forecast.models.tier1_source import (
    Tier1Artifacts,
    Tier1Config,
    predict_mw,
    train_source_model,
)
from carbon_forecast.models.tier2_cnnlstm import E3_TARGET, Tier2Artifacts, Tier2Config


@dataclass
class E3Config:
    tier1: Tier1Config = field(default_factory=Tier1Config)          # final source
    tier1_fold: Tier1Config = field(default_factory=Tier1Config)     # OOF source
    flow: Tier1FlowConfig = field(default_factory=Tier1FlowConfig)   # final flow
    flow_fold: Tier1FlowConfig = field(default_factory=Tier1FlowConfig)  # OOF flow
    tier2: Tier2Config = field(default_factory=lambda: Tier2Config(target_col=E3_TARGET))


@dataclass
class E3Artifacts:
    zone: str
    final_source_models: dict[str, Tier1Artifacts]
    final_flow_models: dict[str, Tier1FlowArtifacts]
    tier2: Tier2Artifacts
    normalizer: Normalizer
    source_cols: list[str]
    flow_cols: list[str]
    partner_ci_cols: list[str]
    dynamic_cols: list[str]
    config: E3Config


# --- column helpers -------------------------------------------------------

def source_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith("prod_") and c.endswith("_mw")]


def flow_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith("net_flow_") and c.endswith("_mw")]


def partner_ci_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith("partner_ci_")]


def _src_name(col: str) -> str:
    return col[len("prod_"):-len("_mw")]


def _flow_partner(col: str) -> str:
    return col[len("net_flow_"):-len("_mw")]


# --- forecast plumbing ----------------------------------------------------

def _matrix_source(art: Tier1Artifacts, frame: pd.DataFrame) -> pd.DataFrame:
    preds, origins = predict_mw(art, frame)
    return pd.DataFrame(preds, index=origins)


def _matrix_flow(art: Tier1FlowArtifacts, frame: pd.DataFrame) -> pd.DataFrame:
    preds, origins = predict_flow_mw(art, frame)
    return pd.DataFrame(preds, index=origins)


def _tier2_origins(frame_norm: pd.DataFrame, cfg: Tier2Config, stride: int) -> pd.DatetimeIndex:
    """Origins Tier 2 windowing produces (all feature channels are NaN-free, so
    windowing the target alone yields the same origins as the full sequence)."""
    t = cfg.target_col
    ds = make_windows(
        frame_norm, history_cols=[t], target_cols=[t],
        lookback=cfg.lookback, horizon=cfg.horizon, stride=stride, drop_na=True,
    )
    return ds.origins


def _oof(
    train_frame: pd.DataFrame,
    items: list[tuple[str, str]],
    train_fn: Callable[[pd.DataFrame, str], object],
    predict_fn: Callable[[object, pd.DataFrame], pd.DataFrame],
    cfg: Tier2Config,
    verbose: int,
) -> dict[str, pd.DataFrame]:
    """Stitched leave-one-year-out forecasts per column over the train period.

    items: (column, model_name) pairs; column keys the output (so its normalizer
    stats apply), model_name is what the train/predict callables consume.
    """
    years = sorted(train_frame.index.year.unique())
    out: dict[str, pd.DataFrame] = {}
    for col, name in items:
        parts = []
        for y in years:
            others = train_frame[train_frame.index.year != y]
            art = train_fn(others, name)
            lo = pd.Timestamp(f"{y}-01-01", tz="UTC") - pd.Timedelta(hours=cfg.lookback)
            hi = pd.Timestamp(f"{y}-12-31 23:00", tz="UTC") + pd.Timedelta(hours=cfg.horizon)
            mat = predict_fn(art, train_frame.loc[lo:hi])
            parts.append(mat[mat.index.year == y])
        out[col] = pd.concat(parts).sort_index()
        if verbose:
            print(f"  OOF {col}: {len(out[col])} origins across {len(years)} folds")
    return out


def _to_dynamic(
    matrices: dict[str, pd.DataFrame],
    cols: list[str],
    origins: pd.DatetimeIndex,
    normalizer: Normalizer,
) -> np.ndarray:
    """Assemble a normalized (N, horizon, len(cols)) tensor aligned to origins,
    standardizing each column's MW forecast with the Tier 2 normalizer stats."""
    arrs = []
    for col in cols:
        vals = matrices[col].reindex(origins).to_numpy(dtype=np.float32)
        arrs.append((vals - normalizer.mean[col]) / normalizer.std[col])
    return np.stack(arrs, axis=-1).astype(np.float32)


def _persistence(
    frame_norm: pd.DataFrame, cols: list[str], origins: pd.DatetimeIndex, horizon: int
) -> np.ndarray:
    """Persistence forecast (last observed value held flat) for the future block,
    in normalized space. The last observed hour at origin t is t-1."""
    prev = origins - pd.Timedelta(hours=1)
    arrs = []
    for col in cols:
        last = frame_norm[col].reindex(prev).to_numpy(dtype=np.float32)
        arrs.append(np.repeat(last[:, None], horizon, axis=1))
    return np.stack(arrs, axis=-1).astype(np.float32)


# --- training -------------------------------------------------------------

def train_e3(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    zone: str,
    cfg: E3Config | None = None,
    verbose: int = 1,
) -> E3Artifacts:
    cfg = cfg or E3Config()
    source_cols = source_columns(train_frame)
    flow_cols = flow_columns(train_frame)
    partner_ci_cols = partner_ci_columns(train_frame)
    dynamic_cols = source_cols + flow_cols + partner_ci_cols
    cfg.tier2.target_col = E3_TARGET
    cfg.tier2.dynamic_cols = dynamic_cols
    if verbose:
        print(f"[E3:{zone}] {len(source_cols)} sources, {len(flow_cols)} flows, "
              f"{len(partner_ci_cols)} partner-CI -> target {E3_TARGET}")

    # Job 2: final Tier 1 models (full train, early-stopped on val).
    if verbose:
        print(f"[E3:{zone}] training final Tier 1 (source + flow)")
    final_source = {
        _src_name(c): train_source_model(train_frame, val_frame, _src_name(c), cfg.tier1, 0)
        for c in source_cols
    }
    final_flow = {
        _flow_partner(c): train_flow_model(train_frame, val_frame, _flow_partner(c), cfg.flow, 0)
        for c in flow_cols
    }

    # Job 1: out-of-fold Tier 1 forecasts (source + flow) over the train period.
    if verbose:
        print(f"[E3:{zone}] generating out-of-fold Tier 1 forecasts")
    oof = _oof(
        train_frame, [(c, _src_name(c)) for c in source_cols],
        lambda fr, s: train_source_model(fr, None, s, cfg.tier1_fold, 0),
        _matrix_source, cfg.tier2, verbose,
    )
    oof.update(_oof(
        train_frame, [(c, _flow_partner(c)) for c in flow_cols],
        lambda fr, p: train_flow_model(fr, None, p, cfg.flow_fold, 0),
        _matrix_flow, cfg.tier2, verbose,
    ))

    # Tier 2 features. The overridden (forecast) channels are sources + flows;
    # partner CI is left to the frame's actuals (strategy A) via override-first-K.
    forecast_cols = source_cols + flow_cols
    t2norm = Normalizer.fit(train_frame)

    tr_origins = _tier2_origins(t2norm.transform(train_frame), cfg.tier2, cfg.tier2.stride)
    fs_train = _to_dynamic(oof, forecast_cols, tr_origins, t2norm)

    val_mat = {c: _matrix_source(final_source[_src_name(c)], val_frame) for c in source_cols}
    val_mat.update({c: _matrix_flow(final_flow[_flow_partner(c)], val_frame) for c in flow_cols})
    va_origins = _tier2_origins(t2norm.transform(val_frame), cfg.tier2, cfg.tier2.val_stride)
    fs_val = _to_dynamic(val_mat, forecast_cols, va_origins, t2norm)

    if verbose:
        print(f"[E3:{zone}] training Tier 2 (train fs {fs_train.shape}, val fs {fs_val.shape})")
    tier2 = t2.train_tier2(
        train_frame, val_frame, cfg.tier2, normalizer=t2norm,
        train_future_dynamic=fs_train, val_future_dynamic=fs_val,
        verbose=2 if verbose else 0,
    )
    return E3Artifacts(
        zone=zone, final_source_models=final_source, final_flow_models=final_flow,
        tier2=tier2, normalizer=t2norm, source_cols=source_cols, flow_cols=flow_cols,
        partner_ci_cols=partner_ci_cols, dynamic_cols=dynamic_cols, config=cfg,
    )


# --- inference ------------------------------------------------------------

def _inference_future_dynamic(art: E3Artifacts, frame: pd.DataFrame) -> np.ndarray:
    """Full dynamic future tensor for inference: source + flow final forecasts,
    then partner-CI persistence (no leakage), in dynamic-column order."""
    origins = _tier2_origins(art.normalizer.transform(frame), art.config.tier2, 1)
    mats = {c: _matrix_source(art.final_source_models[_src_name(c)], frame) for c in art.source_cols}
    mats.update({c: _matrix_flow(art.final_flow_models[_flow_partner(c)], frame) for c in art.flow_cols})
    fs = _to_dynamic(mats, art.source_cols + art.flow_cols, origins, art.normalizer)
    if art.partner_ci_cols:
        frame_n = art.normalizer.transform(frame)
        pers = _persistence(frame_n, art.partner_ci_cols, origins, art.config.tier2.horizon)
        fs = np.concatenate([fs, pers], axis=-1)
    return fs


def predict_ci(art: E3Artifacts, frame: pd.DataFrame) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Full chain: final Tier 1 (source + flow) -> Tier 2 -> 96h consumption-based CI."""
    fs = _inference_future_dynamic(art, frame)
    return t2.predict_ci(art.tier2, frame, future_dynamic=fs)


def evaluate_ci(art: E3Artifacts, frame: pd.DataFrame) -> dict[str, float]:
    """End-to-end E3 accuracy: MAPE (primary), MAE, RMSE on consumption-based CI."""
    fs = _inference_future_dynamic(art, frame)
    return t2.evaluate_ci(art.tier2, frame, future_dynamic=fs)


def predict_with_truth(
    art: E3Artifacts, frame: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Return (preds, y_true, origins), each (N, 96) in gCO2eq/kWh, for
    per-horizon analysis (degradation curves, failure modes)."""
    fs = _inference_future_dynamic(art, frame)
    preds, origins = t2.predict_ci(art.tier2, frame, future_dynamic=fs)
    frame_n = art.tier2.normalizer.transform(frame)
    _, y_norm, _, _ = t2.assemble_sequences(frame_n, art.tier2.config, 1, fs)
    y_true = art.tier2.normalizer.inverse_transform(y_norm, art.tier2.target_col)
    return preds, y_true, origins
