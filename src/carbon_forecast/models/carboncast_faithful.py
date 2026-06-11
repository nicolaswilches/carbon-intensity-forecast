"""E2 orchestrator: CarbonCast-faithful production-based CI model.

Wires the two tiers into one trainable, reproducible pipeline per zone:

  Job 1 (out-of-fold features for Tier 2 training)
    For each train year Y, train a Tier 1 source model on the OTHER years and
    predict Y. Stitch the held-out years into honest, test-like source forecasts
    spanning the whole train period. These feed Tier 2's training so it does not
    over-trust optimistic in-sample forecasts.

  Job 2 (final models for inference)
    Train one Tier 1 source model per source on the FULL train period. These
    produce source forecasts for validation, Test A, and Test B.

  Tier 2
    Train the CNN-LSTM on the OOF forecasts (train) and final-model forecasts
    (val). At inference the chain is always final Tier 1 -> Tier 2.

Only the final Tier 1 models and the trained Tier 2 survive to inference; the
fold models are scaffolding, discarded after Job 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import keras
import numpy as np
import pandas as pd

from carbon_forecast.data.normalize import Normalizer
from carbon_forecast.data.windowing import make_windows
from carbon_forecast.models import tier2_cnnlstm as t2
from carbon_forecast.models.tier1_source import (
    Tier1Artifacts,
    Tier1Config,
    predict_mw,
    train_source_model,
)
from carbon_forecast.models.tier2_cnnlstm import E2_TARGET, Tier2Artifacts, Tier2Config


@dataclass
class E2Config:
    tier1: Tier1Config = field(default_factory=Tier1Config)       # final models
    tier1_fold: Tier1Config = field(default_factory=Tier1Config)  # OOF fold models
    tier2: Tier2Config = field(default_factory=Tier2Config)


@dataclass
class E2Artifacts:
    zone: str
    final_models: dict[str, Tier1Artifacts]
    tier2: Tier2Artifacts
    normalizer: Normalizer
    source_cols: list[str]
    config: E2Config


def source_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith("prod_") and c.endswith("_mw")]


def _src_name(col: str) -> str:
    return col[len("prod_"):-len("_mw")]


def _tier2_origins(frame_norm: pd.DataFrame, cfg: Tier2Config, stride: int) -> pd.DatetimeIndex:
    """Origins Tier 2 windowing will produce (CI/source/weather/calendar are
    NaN-free, so windowing the target alone yields the same origins)."""
    ds = make_windows(
        frame_norm, history_cols=[E2_TARGET], target_cols=[E2_TARGET],
        lookback=cfg.lookback, horizon=cfg.horizon, stride=stride, drop_na=True,
    )
    return ds.origins


def _forecast_matrix(art: Tier1Artifacts, frame: pd.DataFrame) -> pd.DataFrame:
    """Run a Tier 1 model over a frame -> DataFrame indexed by origin, 96 cols (MW)."""
    preds, origins = predict_mw(art, frame)
    return pd.DataFrame(preds, index=origins)


def _oof_matrices(
    train_frame: pd.DataFrame, source_cols: list[str], cfg: E2Config, verbose: int
) -> dict[str, pd.DataFrame]:
    """Per source, stitched out-of-fold forecasts over the train period (MW)."""
    years = sorted(train_frame.index.year.unique())
    out: dict[str, pd.DataFrame] = {}
    for col in source_cols:
        s = _src_name(col)
        parts = []
        for y in years:
            others = train_frame[train_frame.index.year != y]
            fold_art = train_source_model(others, None, s, cfg.tier1_fold, verbose=0)
            # Predict the held-out year on a contiguous slice with 168h history margin.
            lo = pd.Timestamp(f"{y}-01-01", tz="UTC") - pd.Timedelta(hours=cfg.tier2.lookback)
            hi = pd.Timestamp(f"{y}-12-31 23:00", tz="UTC") + pd.Timedelta(hours=cfg.tier2.horizon)
            sl = train_frame.loc[lo:hi]
            mat = _forecast_matrix(fold_art, sl)
            parts.append(mat[mat.index.year == y])
        out[s] = pd.concat(parts).sort_index()
        if verbose:
            print(f"  OOF {col}: {len(out[s])} origins across {len(years)} folds")
    return out


def _build_future_source(
    matrices: dict[str, pd.DataFrame],
    source_cols: list[str],
    origins: pd.DatetimeIndex,
    normalizer: Normalizer,
    horizon: int,
) -> np.ndarray:
    """Assemble normalized (N, horizon, n_sources) future-source tensor aligned
    to `origins`, ordered by source_cols. MW forecasts are standardized with the
    Tier 2 normalizer's per-source stats to match the normalized frame."""
    arrs = []
    for col in source_cols:
        m = matrices[_src_name(col)].reindex(origins)
        vals = m.to_numpy(dtype=np.float32)
        arrs.append((vals - normalizer.mean[col]) / normalizer.std[col])
    return np.stack(arrs, axis=-1).astype(np.float32)


def train_e2(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    zone: str,
    cfg: E2Config | None = None,
    verbose: int = 1,
    seed: int | None = 0,
) -> E2Artifacts:
    # Seed Python/NumPy/TF once so the full Tier 1 -> Tier 2 sequence is reproducible.
    if seed is not None:
        keras.utils.set_random_seed(seed)
    cfg = cfg or E2Config()
    source_cols = source_columns(train_frame)
    if verbose:
        print(f"[E2:{zone}] {len(source_cols)} sources: {[_src_name(c) for c in source_cols]}")

    # Job 2: final Tier 1 models (full train, early-stopped on val).
    if verbose:
        print(f"[E2:{zone}] training {len(source_cols)} final Tier 1 models")
    final_models = {
        _src_name(c): train_source_model(train_frame, val_frame, _src_name(c), cfg.tier1, verbose=0)
        for c in source_cols
    }

    # Job 1: out-of-fold Tier 1 forecasts over the train period.
    if verbose:
        print(f"[E2:{zone}] generating out-of-fold Tier 1 forecasts")
    oof = _oof_matrices(train_frame, source_cols, cfg, verbose)

    # Tier 2 features: OOF for train, final-model forecasts for val.
    t2norm = Normalizer.fit(train_frame)
    tr_origins = _tier2_origins(t2norm.transform(train_frame), cfg.tier2, cfg.tier2.stride)
    fs_train = _build_future_source(oof, source_cols, tr_origins, t2norm, cfg.tier2.horizon)

    val_matrices = {_src_name(c): _forecast_matrix(final_models[_src_name(c)], val_frame)
                    for c in source_cols}
    va_origins = _tier2_origins(t2norm.transform(val_frame), cfg.tier2, cfg.tier2.val_stride)
    fs_val = _build_future_source(val_matrices, source_cols, va_origins, t2norm, cfg.tier2.horizon)

    if verbose:
        print(f"[E2:{zone}] training Tier 2 (train fs {fs_train.shape}, val fs {fs_val.shape})")
    tier2 = t2.train_tier2(
        train_frame, val_frame, cfg.tier2, normalizer=t2norm,
        train_future_dynamic=fs_train, val_future_dynamic=fs_val,
        verbose=2 if verbose else 0,
    )
    return E2Artifacts(
        zone=zone, final_models=final_models, tier2=tier2,
        normalizer=t2norm, source_cols=source_cols, config=cfg,
    )


def _inference_future_source(art: E2Artifacts, frame: pd.DataFrame, stride: int) -> np.ndarray:
    origins = _tier2_origins(art.normalizer.transform(frame), art.config.tier2, stride)
    matrices = {_src_name(c): _forecast_matrix(art.final_models[_src_name(c)], frame)
                for c in art.source_cols}
    return _build_future_source(matrices, art.source_cols, origins, art.normalizer,
                                art.config.tier2.horizon)


def predict_ci(art: E2Artifacts, frame: pd.DataFrame) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Full chain: final Tier 1 -> Tier 2 -> 96h production-based CI (gCO2eq/kWh)."""
    fs = _inference_future_source(art, frame, stride=1)
    return t2.predict_ci(art.tier2, frame, future_dynamic=fs)


def evaluate_ci(art: E2Artifacts, frame: pd.DataFrame) -> dict[str, float]:
    """End-to-end E2 accuracy: MAPE (primary), MAE, RMSE on production-based CI."""
    fs = _inference_future_source(art, frame, stride=1)
    return t2.evaluate_ci(art.tier2, frame, future_dynamic=fs)


def predict_with_truth(
    art: E2Artifacts, frame: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Return (preds, y_true, origins), each (N, 96) in gCO2eq/kWh, for
    per-horizon analysis (degradation curves, failure modes)."""
    fs = _inference_future_source(art, frame, stride=1)
    preds, origins = t2.predict_ci(art.tier2, frame, future_dynamic=fs)
    frame_n = art.tier2.normalizer.transform(frame)
    _, y_norm, _, _ = t2.assemble_sequences(frame_n, art.tier2.config, 1, fs)
    y_true = art.tier2.normalizer.inverse_transform(y_norm, art.tier2.target_col)
    return preds, y_true, origins
