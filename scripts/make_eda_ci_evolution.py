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

import numpy as np
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
# Pad the time axis so the start/end value labels sit in empty space, not on the axis.
XPAD = ("2020-08-01", "2026-11-01")


def _alpha(hex_c: str, a: float) -> str:
    r, g, b = (int(hex_c.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"


def build() -> None:
    fig = go.Figure()
    for z in ZONES:
        df = pd.read_parquet(os.path.join(PROC, f"{z}.parquet"))
        m = df["cons_based_ci"].resample("MS").mean()
        # ISO strings so kaleido's serializer never sees a pandas Timestamp.
        x = m.index.strftime("%Y-%m-%d").tolist()
        y = m.values
        color = P.REGIONAL_PALETTE[z]
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=LABEL[z],
                                 line=dict(color=color, width=1.6)))
        # Linear trend, dotted, same color at 50% opacity.
        t = np.arange(len(y))
        slope, intercept = np.polyfit(t, y, 1)
        fig.add_trace(go.Scatter(x=[x[0], x[-1]], y=[intercept, intercept + slope * t[-1]],
                                 mode="lines", showlegend=False, hoverinfo="skip",
                                 line=dict(color=_alpha(color, 0.5), width=1.3, dash="dot")))
        # Start (2021) and end (2026) carbon-intensity value labels.
        fig.add_annotation(x=x[0], y=y[0], text=f"{y[0]:.0f}", showarrow=False,
                           xanchor="right", xshift=-3,
                           font=dict(size=P.REPORT_FONT - 2, color=color))
        fig.add_annotation(x=x[-1], y=y[-1], text=f"{y[-1]:.0f}", showarrow=False,
                           xanchor="left", xshift=3,
                           font=dict(size=P.REPORT_FONT - 2, color=color))
    P.style_report_fig(fig, span="column", height=440, legend=True, ylabel="gCO₂eq/kWh")
    fig.update_yaxes(rangemode="tozero")
    fig.update_xaxes(range=XPAD)
    out = os.path.join(FIGS, "eda_ci_evolution.pdf")
    fig.write_image(out)
    print("wrote", out)


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    build()
