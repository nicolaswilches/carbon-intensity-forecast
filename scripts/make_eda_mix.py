"""Generate the EDA generation-mix figure (May 2021 vs May 2026).

For each region, two horizontal 100% stacked bars: the top bar is the 2021
generation mix, the bottom bar 2025 (annual means; single months are unreliable
because reactors go fully offline for maintenance, e.g. Belgian nuclear reads 0%
in May 2026). Each segment is one energy source, labeled with its share; sources
below 1% are folded into "other". Five regions stacked vertically, single column.

    .venv/bin/python scripts/make_eda_mix.py
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
Y0, Y1 = "2021", "2025"                # annual means (2025 = latest full year)
BARS = ["2021", "2025"]
OTHER = "other"
OTHER_COLOR = "#D9D4D4"
MIN_SHARE = 1.0                         # below this (both months) -> "other"


def shares(zone: str) -> tuple[dict, dict]:
    df = pd.read_parquet(os.path.join(PROC, f"{zone}.parquet"))
    src = [c for c in df.columns if c.startswith("prod_") and c.endswith("_mw")]
    out = []
    for mo in (Y0, Y1):
        m = df.loc[mo, src].mean()
        tot = m.sum()
        out.append({c.replace("prod_", "").replace("_mw", ""): 100 * m[c] / tot for c in src})
    return out[0], out[1]


def fold(s0: dict, s1: dict) -> tuple[list[str], dict, dict]:
    names = [s for s in P.ENERGY_SOURCE_ORDER
             if max(s0.get(s, 0), s1.get(s, 0)) >= MIN_SHARE]
    kept = set(names)
    a = {s: s0.get(s, 0.0) for s in names}
    b = {s: s1.get(s, 0.0) for s in names}
    o0 = sum(v for k, v in s0.items() if k not in kept)
    o1 = sum(v for k, v in s1.items() if k not in kept)
    if o0 > 0.05 or o1 > 0.05:
        names = names + [OTHER]
        a[OTHER], b[OTHER] = o0, o1
    return names, a, b


def color(src: str) -> str:
    return OTHER_COLOR if src == OTHER else P.ENERGY_PALETTE.get(src, OTHER_COLOR)


def _text_color(hex_c: str) -> str:
    r, g, b = (int(hex_c.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return "#1F1F1F" if lum > 150 else "#FFFFFF"


def build() -> None:
    per_zone = {z: fold(*shares(z)) for z in ZONES}
    shown = {s for names, _, _ in per_zone.values() for s in names}
    order = [s for s in P.ENERGY_SOURCE_ORDER if s in shown]
    if OTHER in shown:
        order.append(OTHER)
    rank = {s: i for i, s in enumerate(order)}

    fig = make_subplots(rows=len(ZONES), cols=1, vertical_spacing=0.085,
                        subplot_titles=[LABEL[z] for z in ZONES])
    seen: set[str] = set()
    for i, z in enumerate(ZONES, start=1):
        names, a, b = per_zone[z]
        for s in names:
            vals = [a[s], b[s]]
            c = color(s)
            first = s not in seen
            seen.add(s)
            fig.add_trace(go.Bar(
                y=BARS, x=vals, orientation="h", legendgroup=s, legendrank=rank[s],
                name=s.replace("_", " ").capitalize(), marker_color=c,
                marker_line=dict(color="#FFFFFF", width=0.7),
                text=[f"{v:.0f}" for v in vals], textposition="inside",
                insidetextanchor="middle", textfont=dict(color=_text_color(c)),
                showlegend=first), row=i, col=1)
        fig.update_yaxes(autorange="reversed", row=i, col=1)  # May 2021 on top

    fig.update_layout(barmode="stack", bargap=0.15,
                      uniformtext=dict(minsize=8, mode="hide"))
    P.style_report_fig(fig, span="column", height=504, legend=True)
    fig.update_layout(margin=dict(b=9))
    fig.update_xaxes(range=[0, 100], showticklabels=False, showgrid=False)
    fig.update_yaxes(tickfont=dict(size=P.REPORT_FONT - 1))
    fig.update_layout(legend=dict(orientation="h", x=0.5, xanchor="center",
                                  y=1.04, yanchor="bottom", font=dict(size=P.REPORT_FONT - 1),
                                  itemwidth=30))
    out = os.path.join(FIGS, "eda_mix.pdf")
    fig.write_image(out)
    print("wrote", out)


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    build()
