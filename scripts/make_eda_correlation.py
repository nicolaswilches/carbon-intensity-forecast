"""Generate the EDA feature-correlation heatmap, sized for a single report column.

Renders the Pearson correlation of each feature with consumption-based carbon
intensity, per zone. Values are the ones reported in the correlation table in the
report (computed in notebook S03); the heatmap is the table's visual twin, so it is
built from those values to stay exactly consistent with the table. Diverging scale,
blue (negative) to red (positive), white at zero.

    .venv/bin/python scripts/make_eda_correlation.py
"""
from __future__ import annotations

import os
import sys

import plotly.graph_objects as go

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from carbon_forecast.plotting import config as P  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIGS = os.path.join(ROOT, "outputs", "figures")

ZONES = ["BE", "FI", "SG", "PJM", "NYIS"]
FEATURES = ["production CI", "partner CI", "renewable share",
            "net import", "total generation", "temperature"]
# rows = features, cols = zones (BE, FI, SG, PJM, NYIS) -- from the report table.
M = [
    [0.86, 0.97, 1.00, 0.99, 0.95],   # production CI
    [0.79, 0.71, 0.38, 0.76, 0.81],   # partner CI
    [-0.66, -0.43, -0.99, -0.53, -0.72],  # renewable share
    [0.20, 0.56, -0.10, -0.21, -0.11],    # net import
    [0.00, -0.02, -0.53, 0.73, 0.68],     # total generation
    [-0.42, -0.58, -0.63, -0.01, 0.22],   # temperature
]
DIVERGING = [[0.0, P.FLOW_EXPORT_COLOR], [0.5, "#FFFFFF"], [1.0, "#C0392B"]]


def build() -> None:
    fig = go.Figure(go.Heatmap(
        z=M, x=ZONES, y=FEATURES, zmid=0, zmin=-1, zmax=1,
        colorscale=DIVERGING, xgap=2, ygap=2,
        colorbar=dict(title="r", thickness=10, len=0.9, tickvals=[-1, 0, 1])))
    for r, row in enumerate(M):
        for c, v in enumerate(row):
            fig.add_annotation(x=ZONES[c], y=FEATURES[r], text=f"{v:.2f}",
                               showarrow=False,
                               font=dict(size=P.REPORT_FONT,
                                         color="#1F1F1F" if abs(v) < 0.6 else "#FFFFFF"))
    P.style_report_fig(fig, span="column", height=520, legend=False)
    fig.update_xaxes(side="top")
    # production CI (first feature) on top; scaleanchor makes the cells exact squares
    fig.update_yaxes(autorange="reversed", scaleanchor="x", scaleratio=1)
    out = os.path.join(FIGS, "eda_correlation.pdf")
    fig.write_image(out)
    print("wrote", out)


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    build()
