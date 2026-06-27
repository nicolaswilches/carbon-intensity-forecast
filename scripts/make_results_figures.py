"""Generate the two Results figures from saved Test A predictions.

Reads the validation-selected E3 consumption-based predictions in
``outputs/preds/{zone}.npz`` (Finland uses the 2023-window model, ``FI_2023.npz``,
matching the report) and writes two report-styled PDFs to ``outputs/figures``:

- ``results_horizon_curve.pdf`` : per-zone error by forecast horizon (1-96 h),
  MAPE and MAE panels. Shows the per-zone degradation pattern.
- ``results_forecast_vs_actual.pdf`` : 24 h-ahead forecast vs actual
  consumption-based carbon intensity over May 2026, one panel per zone.

Run with the project venv to avoid an env rebuild:

    .venv/bin/python scripts/make_results_figures.py
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
# Default to the single-tier consumption model: it is the report's best model, and
# the horizon figures are captioned single-tier. (The two-tier preds live in
# outputs/preds; override with PREDS_SUBDIR if needed.)
PREDS = os.path.join(ROOT, "outputs", os.environ.get("PREDS_SUBDIR", "preds_final_single_cons"))
FIGS = os.path.join(ROOT, "outputs", "figures")

# Display order (best to worst) and the npz file backing each zone. The single-tier
# model uses Finland's full-window file (the 2023-window variant is a two-tier choice).
ZONES = ["SG", "US-NY-NYIS", "US-MIDA-PJM", "FI", "BE"]
NPZ = {z: f"{z}.npz" for z in ZONES}
NPZ["FI"] = os.environ.get("FI_NPZ", "FI.npz")
LABEL = {"SG": "SG", "US-NY-NYIS": "NYIS", "US-MIDA-PJM": "PJM",
         "FI": "FI", "BE": "BE"}


def load(zone: str):
    d = np.load(os.path.join(PREDS, NPZ[zone]), allow_pickle=True)
    return d["preds"], d["y_true"], pd.to_datetime(d["origins"])


def _horizon_one(metric: str, ylabel: str, out_name: str) -> None:
    """One single-panel horizon curve (MAPE or MAE), legend above the plot area."""
    fig = go.Figure()
    for z in ZONES:
        preds, y, _ = load(z)
        ae = np.abs(preds - y)
        if metric == "mape":
            yv = np.nanmean(ae / np.clip(np.abs(y), 1e-6, None), axis=0) * 100
        else:
            yv = np.nanmean(ae, axis=0)
        hours = np.arange(1, preds.shape[1] + 1)
        fig.add_trace(go.Scatter(x=hours, y=yv, mode="lines", name=LABEL[z],
                                 line=dict(color=P.REGIONAL_PALETTE[z], width=P.LINE_WIDTH)))
    for h in (24, 48, 72):
        fig.add_vline(x=h, line=dict(color="rgba(0,0,0,0.12)", width=1))
    P.style_report_fig(fig, span="column", height=257, legend=True)
    fig.update_xaxes(title_text="forecast horizon (hours ahead)")
    fig.update_yaxes(title_text=ylabel)
    # Horizontal legend above the axes (paper y > 1) so it never overlaps the lines;
    # extra top margin reserves room for it.
    fig.update_layout(margin=dict(t=46, r=14, b=46, l=58),
                      legend=dict(orientation="h", x=0.5, xanchor="center",
                                  y=1.04, yanchor="bottom"))
    out = os.path.join(FIGS, out_name)
    fig.write_image(out)
    print("wrote", out)


def horizon_curve() -> None:
    _horizon_one("mape", "MAPE (%)", "results_horizon_mape.pdf")
    _horizon_one("mae", "MAE (gCO₂eq/kWh)", "results_horizon_mae.pdf")


def forecast_vs_actual(h: int = 24) -> None:
    fig = make_subplots(rows=len(ZONES), cols=1,
                        subplot_titles=[LABEL[z] for z in ZONES], vertical_spacing=0.09)
    for i, z in enumerate(ZONES, start=1):
        preds, y, origins = load(z)
        # ISO strings so kaleido's orjson serializer never sees a pandas Timestamp.
        t = (origins + pd.Timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%S").tolist()
        fig.add_trace(go.Scatter(x=t, y=y[:, h - 1], mode="lines", name="actual",
                                 line=dict(color="#7F7F7F", width=1.0),
                                 showlegend=(i == 1)), row=i, col=1)
        fig.add_trace(go.Scatter(x=t, y=preds[:, h - 1], mode="lines",
                                 name=f"forecast ({h}h-ahead)",
                                 line=dict(color=P.REGIONAL_PALETTE[z], width=1.0),
                                 showlegend=(i == 1)), row=i, col=1)
        fig.update_yaxes(title_text="gCO₂eq/kWh", row=i, col=1)
    P.style_report_fig(fig, span="column", height=840, legend=True)
    out = os.path.join(FIGS, "results_forecast_vs_actual.pdf")
    fig.write_image(out)
    print("wrote", out)


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    horizon_curve()
    forecast_vs_actual()
