"""Generate the EDA net-import figure (monthly, 2021-2026).

One panel per region: monthly net import as a share of consumption, drawn as bars
colored by sign (net import positive = red, net export negative = blue, following
config.py / the S02 descriptive notebook). Consumption is domestic generation plus
net imports. Five regions stacked vertically, single report column.

    .venv/bin/python scripts/make_eda_net_import.py
"""
from __future__ import annotations

import os
import sys

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


def build() -> None:
    fig = make_subplots(rows=len(ZONES), cols=1, shared_xaxes=True,
                        vertical_spacing=0.045, subplot_titles=[LABEL[z] for z in ZONES])
    for i, z in enumerate(ZONES, start=1):
        df = pd.read_parquet(os.path.join(PROC, f"{z}.parquet"))
        ni = df["net_import_total_mw"]
        cons = df["total_generation_mw"] + ni
        share = 100 * ni.resample("MS").mean() / cons.resample("MS").mean()
        x = share.index.strftime("%Y-%m-%d").tolist()
        colors = [P.FLOW_IMPORT_COLOR if v >= 0 else P.FLOW_EXPORT_COLOR for v in share.values]
        fig.add_trace(go.Bar(x=x, y=share.values, marker_color=colors,
                             marker_line_width=0, showlegend=False), row=i, col=1)
        fig.add_hline(y=0, line=dict(color="rgba(0,0,0,0.35)", width=0.8), row=i, col=1)

    # Two-entry legend explaining the sign coloring.
    for name, c in [("net import", P.FLOW_IMPORT_COLOR), ("net export", P.FLOW_EXPORT_COLOR)]:
        fig.add_trace(go.Bar(x=[None], y=[None], marker_color=c, name=name,
                             showlegend=True), row=1, col=1)

    P.style_report_fig(fig, span="column", height=720, legend=True)
    fig.update_yaxes(title_text="net import (% of consumption)", row=3, col=1)
    fig.update_layout(legend=dict(orientation="h", x=0.5, xanchor="center",
                                  y=1.03, yanchor="bottom", font=dict(size=P.REPORT_FONT - 1)))
    out = os.path.join(FIGS, "eda_net_import.pdf")
    fig.write_image(out)
    print("wrote", out)


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    build()
