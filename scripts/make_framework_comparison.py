"""Four-framework comparison figure for the Results section.

Single-tier vs two-tier, for each target (production / consumption-based CI), per
zone, on Test A. Two stacked panels because production and consumption MAPE are on
different target series and are not directly comparable across panels; within a
panel the two bars share a target and are comparable. Values from the per-framework
validation-selected CSVs.

    .venv/bin/python scripts/make_framework_comparison.py
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
OUT = os.path.join(ROOT, "outputs")
FIGS = os.path.join(ROOT, "outputs", "figures")

ZONES = ["SG", "US-NY-NYIS", "US-MIDA-PJM", "FI", "BE"]
LABEL = {"BE": "Belgium", "FI": "Finland", "SG": "Singapore",
         "US-MIDA-PJM": "US-MIDA-PJM", "US-NY-NYIS": "US-NY-NYIS"}
SINGLE_C = "#E05312"   # single-tier
TWOTIER_C = "#129FE0"  # two-tier
# (panel title, single-tier csv, two-tier csv)
PANELS = [("Production-based CI", "single_tier_prod", "e2_realsplit_testA_valsel"),
          ("Consumption-based CI", "single_tier_cons", "e3_realsplit_testA_valsel")]


def _mape(csv: str) -> dict:
    df = pd.read_csv(os.path.join(OUT, f"{csv}.csv")).set_index("zone")
    return {z: float(df.loc[z, "test_mape"]) for z in ZONES}


def build() -> None:
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.16,
                        subplot_titles=[p[0] for p in PANELS])
    x = [LABEL[z] for z in ZONES]
    for i, (_, single_csv, two_csv) in enumerate(PANELS, start=1):
        s, t = _mape(single_csv), _mape(two_csv)
        fig.add_trace(go.Bar(x=x, y=[s[z] for z in ZONES], name="single-tier",
                             marker_color=SINGLE_C, showlegend=(i == 1),
                             text=[f"{s[z]:.1f}" for z in ZONES], textposition="outside",
                             textfont=dict(size=P.REPORT_FONT - 3), cliponaxis=False),
                      row=i, col=1)
        fig.add_trace(go.Bar(x=x, y=[t[z] for z in ZONES], name="two-tier",
                             marker_color=TWOTIER_C, showlegend=(i == 1),
                             text=[f"{t[z]:.1f}" for z in ZONES], textposition="outside",
                             textfont=dict(size=P.REPORT_FONT - 3), cliponaxis=False),
                      row=i, col=1)
        fig.update_yaxes(title_text="MAPE (%)", row=i, col=1)
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
    P.style_report_fig(fig, span="full", height=420, legend=True)
    fig.update_layout(legend=dict(orientation="h", x=0.5, xanchor="center",
                                  y=1.06, yanchor="bottom"))
    out = os.path.join(FIGS, "results_framework_comparison.pdf")
    fig.write_image(out)
    print("wrote", out)


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    build()
