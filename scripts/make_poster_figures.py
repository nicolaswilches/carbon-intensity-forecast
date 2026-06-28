"""Generate poster-scale SVG figures for the A0 poster.

Four figures at poster column widths (30/30/40 split, 30mm border, 841mm A0):
- Content width: 841 - 60 = 781mm
- Col 1/2 (30%): 234mm ≈ 886px at 96dpi
- Col 3  (40%): 312mm ≈ 1180px at 96dpi

Font and line weights scaled for readability from 1.5m at A0 print size.

    .venv/bin/python scripts/make_poster_figures.py
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

ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROC   = os.path.join(ROOT, "data", "processed")
PREDS  = os.path.join(ROOT, "outputs", "preds_final_single_cons")
OUT    = os.path.join(ROOT, "outputs")
FIGS   = os.path.join(ROOT, "outputs", "figures", "poster")

ZONES = ["SG", "US-NY-NYIS", "US-MIDA-PJM", "FI", "BE"]
LABEL = {"SG": "SG", "US-NY-NYIS": "NYIS", "US-MIDA-PJM": "PJM", "FI": "FI", "BE": "BE"}

# Poster dimensions (96dpi): col 1/2 = 886px, col 3 = 1180px
W_NARROW = 886   # cols 1 and 2
W_WIDE   = 1180  # col 3

FONT     = 20    # body / tick labels
FONT_SM  = 16    # subplot titles, annotations
LINE_W   = 2.4   # data lines

BG       = "#FAF8F5"
GRID_C   = "#E8E4DE"
INK      = "#1A1A1A"
ACTUAL_C = "#9AA0A6"
SINGLE_C = "#E05312"


def _style(fig, width: int, height: int, *, legend: bool = True,
           xlabel: str = "", ylabel: str = "") -> None:
    fig.update_layout(
        width=width, height=height,
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(family="Inter", size=FONT, color=INK),
        margin=dict(t=48, r=24, b=56, l=72),
        legend=dict(
            orientation="h", x=0.5, xanchor="center", y=1.04, yanchor="bottom",
            font=dict(size=FONT), bgcolor="rgba(0,0,0,0)"
        ) if legend else dict(visible=False),
    )
    fig.update_xaxes(
        showgrid=False, linecolor=GRID_C, tickfont=dict(size=FONT_SM),
        title=dict(text=xlabel, font=dict(size=FONT)),
        automargin=True,
    )
    fig.update_yaxes(
        gridcolor=GRID_C, linecolor=GRID_C, tickfont=dict(size=FONT_SM),
        title=dict(text=ylabel, font=dict(size=FONT)),
        automargin=True,
    )


# ── Figure 1: CI evolution ────────────────────────────────────────────────────

def fig1_ci_evolution() -> None:
    XPAD = ("2020-08-01", "2026-11-01")
    START_YSHIFT = {"FI": -14, "US-NY-NYIS": 16}

    def _alpha(hex_c: str, a: float) -> str:
        r, g, b = (int(hex_c.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        return f"rgba({r},{g},{b},{a})"

    fig = go.Figure()
    for z in ZONES:
        df = pd.read_parquet(os.path.join(PROC, f"{z}.parquet"))
        m  = df["cons_based_ci"].resample("MS").mean()
        x  = m.index.strftime("%Y-%m-%d").tolist()
        y  = m.values
        color = P.REGIONAL_PALETTE[z]
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=LABEL[z],
                                 line=dict(color=color, width=LINE_W)))
        t = np.arange(len(y))
        slope, intercept = np.polyfit(t, y, 1)
        fig.add_trace(go.Scatter(
            x=[x[0], x[-1]], y=[intercept, intercept + slope * t[-1]],
            mode="lines", showlegend=False, hoverinfo="skip",
            line=dict(color=_alpha(color, 0.45), width=1.6, dash="dot")))
        fig.add_annotation(x=x[0], y=y[0], text=f"{y[0]:.0f}", showarrow=False,
                           xanchor="right", xshift=-4,
                           yshift=START_YSHIFT.get(z, 0),
                           font=dict(size=FONT_SM, color=color))
        fig.add_annotation(x=x[-1], y=y[-1], text=f"{y[-1]:.0f}", showarrow=False,
                           xanchor="left", xshift=4,
                           font=dict(size=FONT_SM, color=color))

    _style(fig, W_NARROW, 620, ylabel="gCO₂eq/kWh")
    fig.update_yaxes(rangemode="tozero")
    fig.update_xaxes(range=XPAD)
    out = os.path.join(FIGS, "poster_fig1_ci_evolution.svg")
    fig.write_image(out)
    print("wrote", out)


# ── Figure 10: forecast vs actual (consumption single-tier) ──────────────────

def fig10_fva_cons() -> None:
    H = 24

    def _load(zone: str):
        d = np.load(os.path.join(PREDS, f"{zone}.npz"), allow_pickle=True)
        t = (pd.to_datetime(d["origins"]) + pd.Timedelta(hours=H)).strftime(
            "%Y-%m-%dT%H:%M:%S").tolist()
        return t, d["preds"][:, H - 1], d["y_true"][:, H - 1]

    fig = make_subplots(rows=len(ZONES), cols=1,
                        subplot_titles=[LABEL[z] for z in ZONES],
                        vertical_spacing=0.04)
    first = True
    for r, z in enumerate(ZONES, start=1):
        t, preds, actual = _load(z)
        fig.add_trace(go.Scatter(x=t, y=actual, mode="lines", name="actual",
                      line=dict(color=ACTUAL_C, width=1.4), showlegend=first), row=r, col=1)
        fig.add_trace(go.Scatter(x=t, y=preds, mode="lines", name="single-tier forecast",
                      line=dict(color=SINGLE_C, width=1.4), showlegend=first), row=r, col=1)
        first = False

    _style(fig, W_WIDE, 1100)
    fig.update_layout(margin=dict(t=56, r=24, b=24, l=72))
    fig.update_xaxes(showticklabels=False)
    fig.update_annotations(font_size=FONT_SM)
    fig.update_layout(legend=dict(orientation="h", x=0.5, xanchor="center",
                                  y=1.02, yanchor="bottom", font=dict(size=FONT)))
    out = os.path.join(FIGS, "poster_fig10_fva_cons.svg")
    fig.write_image(out)
    print("wrote", out)


# ── Figure 12: framework comparison, consumption-based ───────────────────────

def fig12_framework_cons() -> None:
    final = pd.read_csv(os.path.join(OUT, "final_metrics.csv"))

    def _mape(fw: str) -> dict:
        df = final[final["framework"] == fw].set_index("zone")
        return {z: float(df.loc[z, "test_mape"]) for z in ZONES}

    s = _mape("single_cons")
    t = _mape("e3_cons")
    x = [LABEL[z] for z in ZONES]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=[s[z] for z in ZONES], name="single-tier",
                         marker_color=SINGLE_C,
                         text=[f"{s[z]:.1f}" for z in ZONES], textposition="outside",
                         textfont=dict(size=FONT_SM)))
    fig.add_trace(go.Bar(x=x, y=[t[z] for z in ZONES], name="two-tier",
                         marker_color="#129FE0",
                         text=[f"{t[z]:.1f}" for z in ZONES], textposition="outside",
                         textfont=dict(size=FONT_SM)))
    fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.0)
    _style(fig, W_WIDE, 480, ylabel="MAPE (%)")
    fig.update_layout(legend=dict(orientation="h", x=0.5, xanchor="center",
                                  y=1.04, yanchor="bottom", font=dict(size=FONT)))
    out = os.path.join(FIGS, "poster_fig12_framework_cons.svg")
    fig.write_image(out)
    print("wrote", out)


# ── Figure 15: Tier 1 horizon error ──────────────────────────────────────────

def fig15_tier1_horizon() -> None:
    csv = os.path.join(OUT, "tier1_horizon_nrmse.csv")
    df  = pd.read_csv(csv, index_col="horizon")

    fig = go.Figure()
    hours = np.arange(1, 97)
    for z in ZONES:
        col = z if z in df.columns else None
        if col is None:
            continue
        fig.add_trace(go.Scatter(x=hours, y=df[col].values, mode="lines",
                                 name=LABEL[z],
                                 line=dict(color=P.REGIONAL_PALETTE[z], width=LINE_W)))
    _style(fig, W_WIDE, 480,
           xlabel="forecast horizon (hours ahead)",
           ylabel="total-gen nRMSE (%)")
    fig.update_xaxes(tickmode="array", tickvals=[24, 48, 72, 96],
                     ticktext=["24", "48", "72", "96"])
    fig.update_layout(legend=dict(orientation="h", x=0.5, xanchor="center",
                                  y=1.04, yanchor="bottom", font=dict(size=FONT)))
    out = os.path.join(FIGS, "poster_fig15_tier1_horizon.svg")
    fig.write_image(out)
    print("wrote", out)


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    fig1_ci_evolution()
    fig10_fva_cons()
    fig12_framework_cons()
    fig15_tier1_horizon()
    print("all poster figures written to", FIGS)
