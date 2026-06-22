"""EDA temporal-pattern heatmaps (single column).

Two figures, each five per-region heatmaps stacked vertically (transposed from the
old side-by-side layout so they fit one report column):

- eda_temporal_hour_dow.pdf: rows = day of week, columns = hour of day (local time).
- eda_temporal_month_year.pdf: rows = year (2021-2026), columns = month. Months with
  no data (Jun-Dec 2026) are left white rather than colored.

Color is each region's own carbon-intensity percentile, so within-region structure
is comparable across regions regardless of absolute level.

    .venv/bin/python scripts/make_eda_temporal.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from carbon_forecast.plotting import config as P  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROC = os.path.join(ROOT, "data", "processed")
FIGS = os.path.join(ROOT, "outputs", "figures")

ZONES = ["BE", "FI", "SG", "US-MIDA-PJM", "US-NY-NYIS"]
LABEL = {"BE": "BE", "FI": "FI", "SG": "SG",
         "US-MIDA-PJM": "PJM", "US-NY-NYIS": "NYIS"}
TZ = {"BE": "Europe/Brussels", "FI": "Europe/Helsinki", "SG": "Asia/Singapore",
      "US-MIDA-PJM": "America/New_York", "US-NY-NYIS": "America/New_York"}
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
CSCALE = P.discrete_percentile_colorscale(n_bins=20)


def _local(z: str) -> pd.Series:
    s = pd.read_parquet(os.path.join(PROC, f"{z}.parquet"))["cons_based_ci"]
    s.index = s.index.tz_convert(TZ[z])
    return s


def _pct(grid: np.ndarray) -> np.ndarray:
    """Per-region percentile of each cell; NaN cells (no data) stay NaN -> white."""
    own = np.sort(grid[~np.isnan(grid)].ravel())
    z = np.searchsorted(own, grid, side="right") / len(own) * 100
    return np.where(np.isnan(grid), np.nan, z)


def _colorbar():
    return dict(orientation="v", thickness=9, len=0.92, x=1.0, xanchor="left",
                tickvals=[0, 50, 100], ticksuffix="%",
                title=dict(text="per-region CI percentile", side="right"),
                tickfont=dict(size=P.REPORT_FONT - 2))


def _style(fig, height):
    P.style_report_fig(fig, span="column", height=height, legend=False)
    # Small base bottom margin; x-axis automargin grows it just enough for the
    # bottom-row tick labels, so no blank band is left under the last panel.
    fig.update_layout(margin=dict(t=24, r=64, b=6, l=44))
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=P.REPORT_FONT - 2))
    fig.update_xaxes(tickfont=dict(size=P.REPORT_FONT - 2))


def hour_dow() -> None:
    fig = make_subplots(rows=len(ZONES), cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, subplot_titles=[LABEL[z] for z in ZONES])
    for i, z in enumerate(ZONES, start=1):
        s = _local(z)
        g = s.groupby([s.index.dayofweek, s.index.hour]).mean().unstack(level=1)
        fig.add_trace(go.Heatmap(
            z=_pct(g.values), x=list(g.columns), y=[DOW[d] for d in g.index],
            colorscale=CSCALE, zmin=0, zmax=100, xgap=0, ygap=0,
            showscale=(i == len(ZONES)), colorbar=_colorbar()), row=i, col=1)
    fig.update_xaxes(tickmode="array", tickvals=[0, 6, 12, 18, 23],
                     ticktext=["00", "06", "12", "18", "23"], row=len(ZONES), col=1)
    _style(fig, height=513)
    out = os.path.join(FIGS, "eda_temporal_hour_dow.pdf")
    fig.write_image(out)
    print("wrote", out)


def month_year() -> None:
    fig = make_subplots(rows=len(ZONES), cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, subplot_titles=[LABEL[z] for z in ZONES])
    for i, z in enumerate(ZONES, start=1):
        # UTC here: month/year is tz-insensitive at monthly scale, and tz_convert
        # would leak a few edge hours into spurious 2020/extra-month cells.
        s = pd.read_parquet(os.path.join(PROC, f"{z}.parquet"))["cons_based_ci"]
        g = s.groupby([s.index.year, s.index.month]).mean().unstack(level=1)
        g = g.reindex(index=range(2021, 2027), columns=range(1, 13))
        fig.add_trace(go.Heatmap(
            z=_pct(g.values), x=MONTHS, y=[str(y) for y in g.index],
            colorscale=CSCALE, zmin=0, zmax=100, xgap=0, ygap=0,
            showscale=(i == len(ZONES)), colorbar=_colorbar()), row=i, col=1)
    _style(fig, height=513)  # same height as hour_dow so both fit one column
    out = os.path.join(FIGS, "eda_temporal_month_year.pdf")
    fig.write_image(out)
    print("wrote", out)


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    hour_dow()
    month_year()
