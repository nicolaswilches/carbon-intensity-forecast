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
PREDS = os.path.join(ROOT, "outputs", "preds")
FIGS = os.path.join(ROOT, "outputs", "figures")

# Display order (best to worst) and the npz file backing each zone. Finland uses
# the shortened 2023-window model, consistent with the report's Finland choice.
ZONES = ["SG", "US-NY-NYIS", "US-MIDA-PJM", "FI", "BE"]
NPZ = {z: f"{z}.npz" for z in ZONES}
NPZ["FI"] = "FI_2023.npz"
LABEL = {"SG": "Singapore", "US-NY-NYIS": "US-NY-NYIS", "US-MIDA-PJM": "US-MIDA-PJM",
         "FI": "Finland", "BE": "Belgium"}


def load(zone: str):
    d = np.load(os.path.join(PREDS, NPZ[zone]), allow_pickle=True)
    return d["preds"], d["y_true"], pd.to_datetime(d["origins"])


def horizon_curve() -> None:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.10,
                        subplot_titles=("MAPE (%)", "MAE (gCO₂eq/kWh)"))
    for z in ZONES:
        preds, y, _ = load(z)
        ae = np.abs(preds - y)
        mape_h = np.nanmean(ae / np.clip(np.abs(y), 1e-6, None), axis=0) * 100
        mae_h = np.nanmean(ae, axis=0)
        hours = np.arange(1, preds.shape[1] + 1)
        color = P.REGIONAL_PALETTE[z]
        fig.add_trace(go.Scatter(x=hours, y=mape_h, mode="lines", name=LABEL[z],
                                 legendgroup=z, line=dict(color=color, width=P.LINE_WIDTH)),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=hours, y=mae_h, mode="lines", name=LABEL[z],
                                 legendgroup=z, showlegend=False,
                                 line=dict(color=color, width=P.LINE_WIDTH)),
                      row=2, col=1)
    for r in (1, 2):
        for h in (24, 48, 72):
            fig.add_vline(x=h, line=dict(color="rgba(0,0,0,0.12)", width=1), row=r, col=1)
    P.style_report_fig(fig, span="column", height=360, legend=True)
    fig.update_xaxes(title_text="forecast horizon (hours ahead)", row=2, col=1)
    out = os.path.join(FIGS, "results_horizon_curve.pdf")
    fig.write_image(out)
    print("wrote", out)


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
