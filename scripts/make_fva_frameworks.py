"""Day-ahead forecast vs actual (Results), one full-width figure per target.

Two separate figures (production-based and consumption-based), each a 5x1
small-multiple stack (rows = zones). Within each cell the actual is gray and
the single-tier and two-tier forecasts overlay it, at the 24-hour-ahead horizon
over Test A.

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
LABEL = {"BE": "BE", "FI": "FI", "SG": "SG",
         "US-MIDA-PJM": "PJM", "US-NY-NYIS": "NYIS"}
H = 24  # day-ahead horizon (index H-1)
ACTUAL_C, SINGLE_C, TWO_C = "#7F7F7F", "#E05312", "#129FE0"
TARGETS = [
    ("prod", "preds_final_single_prod", "preds_final_e2_prod",  "results_fva_prod.pdf"),
    ("cons", "preds_final_single_cons", "preds_final_e3_cons", "results_fva_cons.pdf"),
]


def _load(subdir: str, zone: str):
    d = np.load(os.path.join(PRED, subdir, f"{zone}.npz"), allow_pickle=True)
    t = (pd.to_datetime(d["origins"]) + pd.Timedelta(hours=H)).strftime("%Y-%m-%dT%H:%M:%S").tolist()
    return t, d["preds"][:, H - 1], d["y_true"][:, H - 1]


def build_one(single_dir: str, two_dir: str, out_name: str) -> None:
    """5x1 figure for one target (production or consumption)."""
    fig = make_subplots(rows=len(ZONES), cols=1,
                        subplot_titles=[LABEL[z] for z in ZONES],
                        vertical_spacing=0.055)
    first = True
    for r, z in enumerate(ZONES, start=1):
        t, single, actual = _load(single_dir, z)
        _, two, _ = _load(two_dir, z)
        fig.add_trace(go.Scatter(x=t, y=actual, mode="lines", name="actual",
                      line=dict(color=ACTUAL_C, width=0.9), showlegend=first), row=r, col=1)
        fig.add_trace(go.Scatter(x=t, y=single, mode="lines", name="single-tier",
                      line=dict(color=SINGLE_C, width=0.9), showlegend=first), row=r, col=1)
        fig.add_trace(go.Scatter(x=t, y=two, mode="lines", name="two-tier",
                      line=dict(color=TWO_C, width=0.9), showlegend=first), row=r, col=1)
        first = False
    P.style_report_fig(fig, span="column", height=560, legend=True)
    fig.update_xaxes(showticklabels=False)
    fig.update_layout(legend=dict(orientation="h", x=0.5, xanchor="center",
                                  y=1.03, yanchor="bottom"))
    out = os.path.join(FIGS, out_name)
    fig.write_image(out)
    print("wrote", out)


def build() -> None:
    for _, single_dir, two_dir, out_name in TARGETS:
        build_one(single_dir, two_dir, out_name)


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    build()
