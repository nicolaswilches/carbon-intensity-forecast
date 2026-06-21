"""Day-ahead forecast vs actual for all four frameworks (Results).

A 5x2 small-multiple grid: rows are zones, the left column is the production-based
target and the right column the consumption-based target. Within each cell the
actual is gray and the single-tier and two-tier forecasts overlay it, at the
24-hour-ahead horizon over Test A. Production and consumption are different series,
so the two columns are not cross-comparable. Replaces the single-framework fva.

    .venv/bin/python scripts/make_fva_frameworks.py
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
PRED = os.path.join(ROOT, "outputs")
FIGS = os.path.join(ROOT, "outputs", "figures")

ZONES = ["SG", "US-NY-NYIS", "US-MIDA-PJM", "FI", "BE"]
LABEL = {"BE": "Belgium", "FI": "Finland", "SG": "Singapore",
         "US-MIDA-PJM": "US-MIDA-PJM", "US-NY-NYIS": "US-NY-NYIS"}
H = 24  # day-ahead horizon (index H-1)
ACTUAL_C, SINGLE_C, TWO_C = "#7F7F7F", "#0072B2", "#E69F00"
# (column title, single-tier preds dir, two-tier preds dir)
COLS = [("Production-based CI", "preds_single_prod", "preds_e2_prod"),
        ("Consumption-based CI", "preds_single_cons", "preds")]


def _load(subdir: str, zone: str):
    # Finland uses its 2023-window model for the consumption two-tier, matching the
    # numbers reported elsewhere; production frames only have the full-window file.
    fn = "FI_2023.npz" if (subdir == "preds" and zone == "FI") else f"{zone}.npz"
    d = np.load(os.path.join(PRED, subdir, fn), allow_pickle=True)
    t = (pd.to_datetime(d["origins"]) + pd.Timedelta(hours=H)).strftime("%Y-%m-%dT%H:%M:%S").tolist()
    return t, d["preds"][:, H - 1], d["y_true"][:, H - 1]


def build() -> None:
    titles = [f"{LABEL[z]} ({'prod' if c == 0 else 'cons'})"
              for z in ZONES for c in range(2)]
    fig = make_subplots(rows=len(ZONES), cols=2, subplot_titles=titles,
                        vertical_spacing=0.045, horizontal_spacing=0.08)
    first = True
    for r, z in enumerate(ZONES, start=1):
        for c, (_, single_dir, two_dir) in enumerate(COLS, start=1):
            t, single, actual = _load(single_dir, z)
            _, two, _ = _load(two_dir, z)
            fig.add_trace(go.Scatter(x=t, y=actual, mode="lines", name="actual",
                          line=dict(color=ACTUAL_C, width=0.9), showlegend=first), row=r, col=c)
            fig.add_trace(go.Scatter(x=t, y=single, mode="lines", name="single-tier",
                          line=dict(color=SINGLE_C, width=0.9), showlegend=first), row=r, col=c)
            fig.add_trace(go.Scatter(x=t, y=two, mode="lines", name="two-tier",
                          line=dict(color=TWO_C, width=0.9), showlegend=first), row=r, col=c)
            first = False
    P.style_report_fig(fig, span="column", height=900, legend=True)
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(legend=dict(orientation="h", x=0.5, xanchor="center",
                                  y=1.03, yanchor="bottom"))
    out = os.path.join(FIGS, "results_fva_frameworks.pdf")
    fig.write_image(out)
    print("wrote", out)


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    build()
