"""Normalizer: train-only standardization with invertible target transforms.

Standardizes features to (x - mean) / std so columns on different scales
(generation in MW, CI in gCO2eq/kWh, shares in [0,1], calendar sin/cos in
[-1,1]) train stably. Two locked disciplines live here:

  - Train-only statistics: fit() computes mean/std on the train fold ONLY;
    transform() applies those exact constants to validation and both test
    folds. Fitting on the full series would leak test information.
  - Per-region, per-feature stats: one Normalizer is fit per zone; each column
    gets its own mean/std (CI scales differ across zones, production scales
    differ across fuels).

Targets are standardized for the RMSE loss, then inverse_transform()s back to
physical units before MAPE/MAE reporting, so the fitted stats are persisted
(save/load) rather than recomputed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Columns that must NOT be standardized: already bounded/encoded signals where
# the raw scale is meaningful (cyclical features, binary flags, day index).
DEFAULT_PASSTHROUGH: tuple[str, ...] = (
    "hour_sin", "hour_cos", "yearhour_sin", "yearhour_cos",
    "day_of_week", "is_weekend", "is_holiday",
)


class Normalizer:
    """Per-feature standardizer fit on a train fold.

    Stats are stored as plain dicts {column: value} so they serialize to JSON
    and survive across sessions for inference-time inverse transforms.
    """

    def __init__(self, mean: dict[str, float], std: dict[str, float]) -> None:
        self.mean = mean
        self.std = std

    @property
    def columns(self) -> list[str]:
        return list(self.mean)

    @classmethod
    def fit(
        cls,
        train_df: pd.DataFrame,
        columns: list[str] | None = None,
        passthrough: tuple[str, ...] = DEFAULT_PASSTHROUGH,
    ) -> "Normalizer":
        """Compute mean/std on the TRAIN fold only.

        Defaults to every numeric column except `passthrough`. A zero or NaN
        std (constant or empty column) is replaced by 1.0 so transform is a
        no-op rather than a divide-by-zero.
        """
        if columns is None:
            numeric = train_df.select_dtypes(include="number").columns
            columns = [c for c in numeric if c not in passthrough]
        mean: dict[str, float] = {}
        std: dict[str, float] = {}
        for c in columns:
            col = train_df[c]
            m = float(col.mean())
            s = float(col.std(ddof=0))
            if not np.isfinite(s) or s == 0.0:
                s = 1.0
            mean[c] = m
            std[c] = s
        return cls(mean, std)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize the fitted columns in-place on a copy; others untouched."""
        out = df.copy()
        for c in self.columns:
            if c in out.columns:
                out[c] = (out[c] - self.mean[c]) / self.std[c]
        return out

    def inverse_transform(
        self, data: pd.DataFrame | pd.Series | np.ndarray, columns: list[str] | str
    ) -> pd.DataFrame | pd.Series | np.ndarray:
        """Map standardized values back to physical units.

        `columns` names which fitted column(s) the data correspond to. For a
        Series or 1-D array pass a single column name (the target); for a
        DataFrame pass the matching list of column names.
        """
        if isinstance(columns, str):
            m, s = self.mean[columns], self.std[columns]
            return data * s + m
        if isinstance(data, pd.DataFrame):
            out = data.copy()
            for c in columns:
                out[c] = out[c] * self.std[c] + self.mean[c]
            return out
        # ndarray with column-aligned last axis
        arr = np.asarray(data, dtype=float)
        means = np.array([self.mean[c] for c in columns])
        stds = np.array([self.std[c] for c in columns])
        return arr * stds + means

    def save(self, path: Path | str) -> Path:
        """Persist stats as JSON for inference-time inverse transforms."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"mean": self.mean, "std": self.std}, indent=2))
        return path

    @classmethod
    def load(cls, path: Path | str) -> "Normalizer":
        payload = json.loads(Path(path).read_text())
        return cls(payload["mean"], payload["std"])
