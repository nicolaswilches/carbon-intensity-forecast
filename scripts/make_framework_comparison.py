"""Single-tier vs two-tier comparison, one single-column figure per target.

Single-tier vs two-tier per zone, on Test A, as grouped bars. One figure for the
production target and one for the consumption target (the two are different series
and not cross-comparable). Each figure is sized for \\columnwidth. Within a region
the two bars touch (no group gap); gaps separate the regions. Values from the
per-framework validation-selected CSVs.

    .venv/bin/python scripts/make_framework_comparison.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from carbon_forecast.plotting import config as P  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "outputs")
FIGS = os.path.join(ROOT, "outputs", "figures")

ZONES = ["SG", "US-NY-NYIS", "US-MIDA-PJM", "FI", "BE"]
LABEL = {"BE": "BE", "FI": "FI", "SG": "SG",
         "US-MIDA-PJM": "PJM", "US-NY-NYIS": "NYIS"}
SINGLE_C = "#E05312"   # single-tier
TWOTIER_C = "#129FE0"  # two-tier
# (single-tier framework, two-tier framework, output file). Values come from the
# per-zone-window final set (full history for SG/NYIS/PJM, 2024 for FI/BE).
TARGETS = [
    ("single_prod", "e2_prod", "results_framework_prod.pdf"),
    ("single_cons", "e3_cons", "results_framework_cons.pdf"),
]

_FINAL = pd.read_csv(os.path.join(OUT, "final_metrics.csv"))


def _mape(framework: str) -> dict:
    df = _FINAL[_FINAL["framework"] == framework].set_index("zone")
    return {z: float(df.loc[z, "test_mape"]) for z in ZONES}


def build_one(single_csv: str, two_csv: str, out_name: str, legend: bool = True) -> None:
    s, t = _mape(single_csv), _mape(two_csv)
    x = [LABEL[z] for z in ZONES]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=[s[z] for z in ZONES], name="single-tier",
                         marker_color=SINGLE_C,
                         text=[f"{s[z]:.1f}" for z in ZONES], textposition="outside",
                         textfont=dict(size=P.REPORT_FONT - 3), cliponaxis=False))
    fig.add_trace(go.Bar(x=x, y=[t[z] for z in ZONES], name="two-tier",
                         marker_color=TWOTIER_C,
                         text=[f"{t[z]:.1f}" for z in ZONES], textposition="outside",
                         textfont=dict(size=P.REPORT_FONT - 3), cliponaxis=False))
    fig.update_yaxes(title_text="MAPE (%)")
    # bargroupgap=0 so the two bars in a region touch; bargap separates regions.
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.0)
    P.style_report_fig(fig, span="column", height=216, legend=legend)
    if legend:
        fig.update_layout(legend=dict(orientation="h", x=0.5, xanchor="center",
                                      y=1.04, yanchor="bottom"))
    out = os.path.join(FIGS, out_name)
    fig.write_image(out)
    print("wrote", out)


def build() -> None:
    build_one("single_prod", "e2_prod",   "results_framework_prod.pdf", legend=True)
    build_one("single_cons", "e3_cons",   "results_framework_cons.pdf", legend=False)


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    build()
