"""Generate the EDA generation-mix transition figure (2021 -> 2025).

For each zone, an alluvial diagram: a left bar (2021 mix shares) and a right bar
(2025 mix shares), with one tapering ribbon per energy source connecting its 2021
block to its 2025 block. Ribbon width tracks the share, so a shrinking source
narrows and a growing source widens. Same source to same source only, so no
cross-flows are invented. Five zones stacked vertically; sized for a single
report column. Colors and source order follow the project palette; legend on top.

    .venv/bin/python scripts/make_eda_mix_sankey.py
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
PROC = os.path.join(ROOT, "data", "processed")
FIGS = os.path.join(ROOT, "outputs", "figures")

ZONES = ["BE", "FI", "SG", "US-MIDA-PJM", "US-NY-NYIS"]
LABEL = {"BE": "Belgium", "FI": "Finland", "SG": "Singapore",
         "US-MIDA-PJM": "US-MIDA-PJM", "US-NY-NYIS": "US-NY-NYIS"}
Y0, Y1 = "2021", "2025"

# Bar geometry in axis coordinates (x in [0,1], y in [0,100] share %).
XL0, XL1 = 0.00, 0.06   # left (2021) bar
XR0, XR1 = 0.94, 1.00   # right (2025) bar
OTHER = "other"
OTHER_COLOR = "#D9D4D4"
MIN_SHARE = 1.0         # sources below this (both years) fold into "other"
LABEL_MIN = 10.0        # annotate a segment only if its share >= this


def shares(zone: str) -> tuple[dict, dict]:
    df = pd.read_parquet(os.path.join(PROC, f"{zone}.parquet"))
    src = [c for c in df.columns if c.startswith("prod_") and c.endswith("_mw")]
    out = []
    for yr in (Y0, Y1):
        m = df.loc[yr, src].mean()
        tot = m.sum()
        out.append({c.replace("prod_", "").replace("_mw", ""): 100 * m[c] / tot for c in src})
    return out[0], out[1]


def fold(s21: dict, s25: dict) -> tuple[list[str], dict, dict]:
    """Order sources by the palette; fold sub-threshold ones into 'other'."""
    names = [s for s in P.ENERGY_SOURCE_ORDER
             if max(s21.get(s, 0), s25.get(s, 0)) >= MIN_SHARE]
    kept = set(names)
    a = {s: s21.get(s, 0.0) for s in names}
    b = {s: s25.get(s, 0.0) for s in names}
    o21 = sum(v for k, v in s21.items() if k not in kept)
    o25 = sum(v for k, v in s25.items() if k not in kept)
    if o21 > 0.05 or o25 > 0.05:
        names = names + [OTHER]
        a[OTHER], b[OTHER] = o21, o25
    return names, a, b


def color(src: str) -> str:
    return OTHER_COLOR if src == OTHER else P.ENERGY_PALETTE.get(src, OTHER_COLOR)


def rect(fig, row, x0, x1, y0, y1, c, name):
    fig.add_trace(go.Scatter(
        x=[x0, x1, x1, x0, x0], y=[y0, y0, y1, y1, y0],
        fill="toself", fillcolor=c, mode="lines",
        line=dict(width=0.6, color="#FFFFFF"),
        hoverinfo="skip", showlegend=False, name=name), row=row, col=1)


def ribbon(fig, row, lt, lb, rt, rb, c):
    t = np.linspace(0, 1, 40)
    w = (1 - np.cos(np.pi * t)) / 2          # smooth S-curve
    x = XL1 + (XR0 - XL1) * t
    top = lt + (rt - lt) * w
    bot = lb + (rb - lb) * w
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([top, bot[::-1]]),
        fill="toself", fillcolor=_alpha(c, 0.70), mode="lines",
        line=dict(width=0), hoverinfo="skip", showlegend=False), row=row, col=1)


def _alpha(hex_c: str, a: float) -> str:
    r, g, b = (int(hex_c.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"


def build() -> None:
    fig = make_subplots(rows=len(ZONES), cols=1,
                        subplot_titles=[LABEL[z] for z in ZONES],
                        vertical_spacing=0.06)
    per_zone = {z: fold(*shares(z)) for z in ZONES}

    # Legend once, in canonical palette order, 'other' last; only sources shown.
    shown = {s for names, _, _ in per_zone.values() for s in names}
    legend_order = [s for s in P.ENERGY_SOURCE_ORDER if s in shown]
    if OTHER in shown:
        legend_order.append(OTHER)
    for s in legend_order:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=9, color=color(s), symbol="square"),
            name=s.replace("_", " ").capitalize(), showlegend=True), row=1, col=1)

    for i, z in enumerate(ZONES, start=1):
        names, a, b = per_zone[z]
        # Stack from the bottom in palette order ('other' last, on top).
        cyl, cyr = 0.0, 0.0  # running cumulative bottoms for left/right bars
        for s in names:
            la, ra = a[s], b[s]
            lb_, lt_ = cyl, cyl + la
            rb_, rt_ = cyr, cyr + ra
            c = color(s)
            ribbon(fig, i, lt_, lb_, rt_, rb_, c)
            rect(fig, i, XL0, XL1, lb_, lt_, c, s)
            rect(fig, i, XR0, XR1, rb_, rt_, c, s)
            # label large segments
            for share, xc, yb, yt in ((la, (XL0 + XL1) / 2, lb_, lt_),
                                      (ra, (XR0 + XR1) / 2, rb_, rt_)):
                if share >= LABEL_MIN:
                    fig.add_annotation(x=xc, y=(yb + yt) / 2, text=f"{share:.0f}",
                                       showarrow=False, font=dict(size=8, color="#FFFFFF"),
                                       row=i, col=1)
            cyl, cyr = lt_, rt_

    P.style_report_fig(fig, span="column", height=1020, legend=True)
    fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False,
                     range=[-0.04, 1.04])
    fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False,
                     range=[0, 100])
    # 2021 / 2025 labels only under the bottom subplot
    fig.update_xaxes(showticklabels=True, tickvals=[(XL0 + XL1) / 2, (XR0 + XR1) / 2],
                     ticktext=["2021", "2025"], row=len(ZONES), col=1)
    fig.update_layout(margin=dict(t=64, r=10, b=28, l=10),
                      legend=dict(orientation="h", x=0.5, xanchor="center",
                                  y=1.045, yanchor="bottom", font=dict(size=P.REPORT_FONT),
                                  itemwidth=30))
    out = os.path.join(FIGS, "eda_mix_sankey.pdf")
    fig.write_image(out)
    print("wrote", out)


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    build()
