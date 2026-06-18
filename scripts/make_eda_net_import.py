"""Generate the EDA net-import figure (monthly mean, 2021-2026).

Overlays each zone's monthly net import as a share of consumption (positive = net
importer, negative = net exporter), on one panel colored by the regional palette.
Consumption is domestic generation plus net imports. Monthly granularity smooths
daily noise while keeping the seasonal import pattern legible in a single report
column. Replaces the earlier annual, full-width version.

    .venv/bin/python scripts/make_eda_net_import.py
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

ZONES = ["BE", "FI", "US-MIDA-PJM", "US-NY-NYIS", "SG"]
LABEL = {"BE": "Belgium", "FI": "Finland", "SG": "Singapore",
         "US-MIDA-PJM": "US-MIDA-PJM", "US-NY-NYIS": "US-NY-NYIS"}


def build() -> None:
    fig = go.Figure()
    for z in ZONES:
        df = pd.read_parquet(os.path.join(PROC, f"{z}.parquet"))
        ni = df["net_import_total_mw"]
        cons = df["total_generation_mw"] + ni
        share = 100 * ni.resample("MS").mean() / cons.resample("MS").mean()
        x = share.index.strftime("%Y-%m-%d").tolist()
        fig.add_trace(go.Scatter(x=x, y=share.values, mode="lines", name=LABEL[z],
                                 line=dict(color=P.REGIONAL_PALETTE[z], width=1.6)))
    fig.add_hline(y=0, line=dict(color="rgba(0,0,0,0.35)", width=1, dash="dot"))
    P.style_report_fig(fig, span="column", height=330, legend=True,
                       ylabel="net import (% of consumption)")
    out = os.path.join(FIGS, "eda_net_import.pdf")
    fig.write_image(out)
    print("wrote", out)


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    build()
