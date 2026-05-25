"""Project-wide plot configuration.

Locked design language (project_thesis.md, with style refinements logged in
memory/feedback_style.md):
- Template: plotly_white
- Font: Arial. Title 18pt bold, axis titles 14pt, ticks 12pt, legend 12pt.
- Grid: light gray horizontal only (rgba(0,0,0,0.1)). No vertical grid.
- Legend: horizontal, anchored to the top-left of the plot area.
- Three categorical palettes: regional (5 zones), energy source
  (CarbonCast-aligned), model comparison.

Usage:

    from carbon_forecast.plotting.config import (
        REGIONAL_PALETTE, ENERGY_PALETTE, MODEL_PALETTE,
        PLOT_W, PLOT_H, apply_defaults, style_fig,
    )
    apply_defaults()                # once per notebook
    fig = px.line(...)
    style_fig(fig, "Title goes here")
    fig.show()
"""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio


PLOT_W: int = 900
PLOT_H: int = 500

GRID_COLOR: str = "rgba(0,0,0,0.1)"
FONT_FAMILY: str = "Arial"

# Layout margins, used to compute the paper-coord left edge that title and
# legend share with the plot area. Single source of truth: change here and
# everything else stays aligned.
MARGIN_L: int = 70
MARGIN_R: int = 40
MARGIN_T: int = 130
MARGIN_B: int = 60

# Paper-coord x of the plot area's left edge. Title.x and legend.x use it
# so all three (title, legend, axes) share a left edge.
PLOT_AREA_LEFT_PAPER: float = MARGIN_L / PLOT_W


REGIONAL_PALETTE: dict[str, str] = {
    "BE":          "#0072B2",  # blue
    "FI":          "#56B4E9",  # sky blue
    "SG":          "#E69F00",  # orange
    "US-MIDA-PJM": "#009E73",  # green
    "US-NY-NYIS":  "#D55E00",  # vermillion
}


ENERGY_PALETTE: dict[str, str] = {
    "gas":               "#F0A000",
    "coal":              "#404040",
    "oil":               "#8B3A1E",
    "nuclear":           "#A52A2A",
    "wind":              "#3B7DD8",
    "solar":             "#F4D03F",
    "hydro":             "#2E8B8B",
    "biomass":           "#3CB371",
    "unknown":           "#B0B0B0",
    "geothermal":        "#7E5C00",
    "hydro_discharge":   "#5FAFAF",
    "battery_discharge": "#9966CC",
}


MODEL_PALETTE: dict[str, str] = {
    "open":                "#0072B2",  # blue
    "em_operational":      "#E69F00",  # orange
    "carboncast_faithful": "#7F7F7F",  # neutral gray
}


def apply_defaults() -> None:
    """Set process-wide plotly defaults. Call once at notebook startup."""
    pio.templates.default = "plotly_white"


def style_fig(fig: go.Figure, title: str) -> go.Figure:
    """Apply the locked layout to a figure. Returns the figure for chaining.

    Title is rendered bold via HTML; do not pre-wrap in `<b>...</b>` —
    this helper does it.
    """
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(family=FONT_FAMILY, size=18),
            x=PLOT_AREA_LEFT_PAPER,
            xanchor="left",
            y=0.92,
            yanchor="top",
        ),
        font=dict(family=FONT_FAMILY, size=12),
        width=PLOT_W,
        height=PLOT_H,
        xaxis=dict(
            title=dict(font=dict(size=14)),
            tickfont=dict(size=12),
            showgrid=False,
        ),
        yaxis=dict(
            title=dict(font=dict(size=14)),
            tickfont=dict(size=12),
            gridcolor=GRID_COLOR,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="left",
            x=PLOT_AREA_LEFT_PAPER,
            font=dict(size=12),
        ),
        margin=dict(t=MARGIN_T, r=MARGIN_R, b=MARGIN_B, l=MARGIN_L),
    )
    return fig
