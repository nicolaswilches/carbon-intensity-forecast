"""EDA carbon-intensity distribution figures (single column).

Two figures for the report's distribution subsection:

- eda_distribution.pdf: overlaid density histograms of consumption-based carbon
  intensity for all five regions (pooled over 2021-2026), the five-region
  comparison.
- eda_distribution_shift.pdf: one panel per region, each overlaying the 2021 and
  2026 distributions with descriptive statistics in the top corners (2021 left,
  2026 right), so the per-region shift over the study period is visible. Both years
  use the January-May window, since 2026 only runs through May, for a fair compare.

    .venv/bin/python scripts/make_eda_distribution.py
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

ZONES = ["SG", "US-MIDA-PJM", "US-NY-NYIS", "BE", "FI"]  # high to low CI
LABEL = {"BE": "Belgium", "FI": "Finland", "SG": "Singapore",
         "US-MIDA-PJM": "US-MIDA-PJM", "US-NY-NYIS": "US-NY-NYIS"}
C2021 = "#9AA0A6"   # gray for the 2021 distribution
MAXMONTH = 5        # Jan-May window, matched across years (2026 ends in May)


def _alpha(hex_c: str, a: float) -> str:
    r, g, b = (int(hex_c.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"


def _series(z: str) -> pd.Series:
    return pd.read_parquet(os.path.join(PROC, f"{z}.parquet"))["cons_based_ci"]


def _stats_text(s: pd.Series, year: int) -> str:
    return (f"{year}<br>μ {s.mean():.0f}<br>med {s.median():.0f}"
            f"<br>σ {s.std():.0f}<br>skew {s.skew():.2f}")


def overlay() -> None:
    fig = go.Figure()
    for z in ZONES:
        s = _series(z)
        fig.add_trace(go.Histogram(
            x=s.values, name=LABEL[z], histnorm="probability density", nbinsx=60,
            marker_color=_alpha(P.REGIONAL_PALETTE[z], 0.55),
            marker_line=dict(width=0)))
    fig.update_layout(barmode="overlay")
    P.style_report_fig(fig, span="column", height=300, legend=True,
                       xlabel="gCO₂eq/kWh", ylabel="density")
    out = os.path.join(FIGS, "eda_distribution.pdf")
    fig.write_image(out)
    print("wrote", out)


def shift() -> None:
    fig = make_subplots(rows=len(ZONES), cols=1, vertical_spacing=0.055,
                        subplot_titles=[LABEL[z] for z in ZONES])
    for i, z in enumerate(ZONES, start=1):
        s = _series(z)
        s21 = s[(s.index.year == 2021) & (s.index.month <= MAXMONTH)]
        s26 = s[(s.index.year == 2026) & (s.index.month <= MAXMONTH)]
        color = P.REGIONAL_PALETTE[z]
        # No legend: each panel's 2026 uses its own region color, so a shared
        # swatch would be misleading. The corner stat labels (gray 2021, colored
        # 2026) carry the key instead.
        fig.add_trace(go.Histogram(x=s21.values, histnorm="probability density", nbinsx=40,
                                   marker_color=_alpha(C2021, 0.6), marker_line=dict(width=0),
                                   showlegend=False), row=i, col=1)
        fig.add_trace(go.Histogram(x=s26.values, histnorm="probability density", nbinsx=40,
                                   marker_color=_alpha(color, 0.55), marker_line=dict(width=0),
                                   showlegend=False), row=i, col=1)
        sfx = "" if i == 1 else str(i)
        fig.add_annotation(xref=f"x{sfx} domain", yref=f"y{sfx} domain",
                           x=0.01, y=0.97, xanchor="left", yanchor="top", align="left",
                           text=_stats_text(s21, 2021), showarrow=False,
                           font=dict(size=P.REPORT_FONT - 3, color="#5F6368"))
        fig.add_annotation(xref=f"x{sfx} domain", yref=f"y{sfx} domain",
                           x=0.99, y=0.97, xanchor="right", yanchor="top", align="right",
                           text=_stats_text(s26, 2026), showarrow=False,
                           font=dict(size=P.REPORT_FONT - 3, color=color))
        fig.update_xaxes(title_text="gCO₂eq/kWh", row=i, col=1)
    fig.update_layout(barmode="overlay")
    P.style_report_fig(fig, span="column", height=900, legend=False)
    fig.update_layout(margin=dict(t=24, r=10, b=46, l=56))
    out = os.path.join(FIGS, "eda_distribution_shift.pdf")
    fig.write_image(out)
    print("wrote", out)
    # stats for the prose
    for z in ZONES:
        s = _series(z)
        a = s[(s.index.year == 2021) & (s.index.month <= MAXMONTH)]
        b = s[(s.index.year == 2026) & (s.index.month <= MAXMONTH)]
        print(f"{z:12s} 2021 mean={a.mean():.0f} std={a.std():.0f} skew={a.skew():.2f}"
              f"  ->  2026 mean={b.mean():.0f} std={b.std():.0f} skew={b.skew():.2f}")


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    overlay()
    shift()
