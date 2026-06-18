"""Generate the EDA carbon-intensity evolution figure (monthly mean, 2021-2026).

Overlays the monthly-mean consumption-based carbon intensity of all five zones on
one panel, colored by the regional palette. Monthly granularity (65 points/zone)
smooths daily noise while keeping the seasonal cycle and long-term trend legible in
a single report column. Replaces the earlier dense daily-mean, full-width version.

    .venv/bin/python scripts/make_eda_ci_evolution.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from carbon_forecast.plotting import config as P  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROC = os.path.join(ROOT, "data", "processed")
FIGS = os.path.join(ROOT, "outputs", "figures")

ZONES = ["SG", "US-MIDA-PJM", "US-NY-NYIS", "BE", "FI"]  # legend order, high to low
LABEL = {"BE": "Belgium", "FI": "Finland", "SG": "Singapore",
         "US-MIDA-PJM": "US-MIDA-PJM", "US-NY-NYIS": "US-NY-NYIS"}


def build() -> None:
    fig = go.Figure()
    for z in ZONES:
        df = pd.read_parquet(os.path.join(PROC, f"{z}.parquet"))
        m = df["cons_based_ci"].resample("MS").mean()
        # ISO strings so kaleido's serializer never sees a pandas Timestamp.
        x = m.index.strftime("%Y-%m-%d").tolist()
        fig.add_trace(go.Scatter(x=x, y=m.values, mode="lines", name=LABEL[z],
                                 line=dict(color=P.REGIONAL_PALETTE[z], width=1.6)))
    P.style_report_fig(fig, span="column", height=330, legend=True,
                       ylabel="gCO₂eq/kWh")
    fig.update_yaxes(rangemode="tozero")
    out = os.path.join(FIGS, "eda_ci_evolution.pdf")
    fig.write_image(out)
    print("wrote", out)


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    build()
