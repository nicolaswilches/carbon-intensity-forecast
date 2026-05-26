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


PLOT_W: int = 1000
PLOT_H: int = 500

GRID_COLOR: str = "rgba(0,0,0,0.1)"
FONT_FAMILY: str = "Arial"

# Shared canvas: warm off-white behind the whole figure and the plot area,
# near-black for all text. Applied by style_fig to every chart.
BG_COLOR: str = "#F7F0E4"
TEXT_COLOR: str = "#1F1F1F"

# Title sits this many pixels below the figure's top edge. Expressed in
# pixels (not a paper fraction) so the buffer between title and plot stays
# constant across figures of different heights.
TITLE_TOP_PAD_PX: int = 28

# Title and subtitle font sizes. Subtitle is 25% smaller than the title and
# not bold. Editable per chart via style_fig(..., subtitle="...").
TITLE_SIZE: int = 20
SUBTITLE_SIZE: int = int(TITLE_SIZE * 0.75)

# Subplot titles (the per-panel labels like "BE") sit flush against the top
# of each panel by default. Lift them this many pixels so they do not crowd
# the chart.
SUBPLOT_TITLE_YSHIFT_PX: int = 12

# Minimum line width for line traces. Plotly draws the legend swatch at the
# trace's line width, so thin lines give invisible legend swatches. style_fig
# bumps any thinner line up to this so legend colors are always legible.
LINE_WIDTH: float = 1.0

# Layout margins, used to compute the paper-coord left edge that title and
# legend share with the plot area. Single source of truth: change here and
# everything else stays aligned.
MARGIN_L: int = 100
MARGIN_R: int = 80
MARGIN_T: int = 120
MARGIN_B: int = 80

# Paper-coord x of the plot area's left edge. Title.x and legend.x use it
# so all three (title, legend, axes) share a left edge.
PLOT_AREA_LEFT_PAPER: float = MARGIN_L / PLOT_W


REGIONAL_PALETTE: dict[str, str] = {
    "BE": "#0072B2",  # blue
    "FI": "#56B4E9",  # sky blue
    "SG": "#E69F00",  # orange
    "US-MIDA-PJM": "#009E73",  # green
    "US-NY-NYIS":  "#D55E00",  # vermillion
}


# Canonical display order for energy sources. Baseload at the bottom of a
# stacked area; continuous color progression cool -> warm. Diverges from
# CarbonCast for legibility; see project_thesis.md amendment 2026-05-25.
ENERGY_SOURCE_ORDER: tuple[str, ...] = (
    "nuclear",
    "battery_discharge",
    "hydro_discharge",
    "hydro",
    "wind",
    "biomass",
    "solar",
    "geothermal",
    "gas",
    "coal",
    "oil",
    "unknown",
)

ENERGY_PALETTE: dict[str, str] = {
    "nuclear": "#493C6B",
    "battery_discharge": "#655391",
    "hydro": "#3E54B3",
    "hydro_discharge": "#59679E",
    "wind": "#60B2F0",
    "biomass": "#C4D61E",
    "solar": "#FFB300",
    "geothermal": "#FF6A00",
    "gas": "#FC7D6A",
    "coal": "#785044",
    "oil": "#2E201D",
    "unknown": "#A69C9C",
}


MODEL_PALETTE: dict[str, str] = {
    "open": "#0072B2",  # blue
    "em_operational": "#E69F00",  # orange
    "carboncast_faithful": "#7F7F7F",  # neutral gray
}

# Carbon intensity is the project's headline series. Use this one color
# everywhere CI is drawn as a line, so it is instantly recognizable across
# charts. Warm orange in the geothermal family but slightly deeper, so it
# stays distinct from geothermal (#FF6A00) if both ever share a chart.
CI_COLOR: str = "#E34C17"

# Net cross-border flow bars are colored by sign: net import (positive) reads
# warm/red, net export (negative) reads cool/blue.
FLOW_IMPORT_COLOR: str = "#C0504D"  # importing, red hue
FLOW_EXPORT_COLOR: str = "#3B6A96"  # exporting, blue hue


def apply_defaults() -> None:
    """Set process-wide plotly defaults. Call once at notebook startup."""
    pio.templates.default = "plotly_white"


def _title_case_word(w: str) -> str:
    """Title-case one whitespace-delimited token, preserving acronyms and units.

    Rules:
    - A token whose alphabetic content has any uppercase letter beyond the
      first is treated as an acronym or unit (UTC, MW, TWh, gCO2eq) and
      returned unchanged.
    - Otherwise, every first letter of an alphabetic run is uppercased
      (so hyphens and parentheses act as word breaks), the rest lowercased.
    """
    if not w:
        return w
    alpha_only = "".join(c for c in w if c.isalpha())
    if alpha_only and any(c.isupper() for c in alpha_only[1:]):
        return w
    out = []
    capitalize_next = True
    for c in w:
        if c.isalpha():
            out.append(c.upper() if capitalize_next else c.lower())
            capitalize_next = False
        else:
            out.append(c)
            capitalize_next = True
    return "".join(out)


def title_case(s: str) -> str:
    """Title-case a free string while preserving acronyms and unit symbols.

    Splits on whitespace and applies `_title_case_word` to each token.
    Underscores are not touched here; callers wanting `snake_case ->
    Title Case` should `.replace("_", " ")` first.
    """
    return " ".join(_title_case_word(w) for w in s.split())


def _finalize_axes(fig: go.Figure) -> None:
    """Style and retitle every axis, and Title-Case every trace name, in place.

    Iterates all axes (xaxis, xaxis2, ... yaxis, yaxis2, ...) so multi-row
    subplots get the same tick font, grid policy, and Title-Cased titles as
    a single-plot figure, not just the first subplot.
    """
    for key in list(fig.layout):
        if key.startswith("xaxis"):
            ax = fig.layout[key]
            ax.update(tickfont=dict(size=12), showgrid=False)
            if ax.title is not None and ax.title.text:
                ax.title.update(text=title_case(ax.title.text), font=dict(size=14))
        elif key.startswith("yaxis"):
            ax = fig.layout[key]
            ax.update(tickfont=dict(size=12), gridcolor=GRID_COLOR)
            if ax.title is not None and ax.title.text:
                ax.title.update(text=title_case(ax.title.text), font=dict(size=14))
    for trace in fig.data:
        name = getattr(trace, "name", None)
        if isinstance(name, str) and name:
            trace.name = title_case(name.replace("_", " "))
        # Thicken thin line swatches so the legend color is visible. Only
        # Scatter traces carry a top-level `.line`; Bar/Heatmap do not.
        if isinstance(trace, go.Scatter) and (trace.mode is None or "lines" in trace.mode):
            if trace.line.width is None or trace.line.width < LINE_WIDTH:
                trace.line.width = LINE_WIDTH
    # Lift subplot titles (make_subplots / facet labels are annotations) off
    # the top of their panels so they do not crowd the chart.
    for ann in fig.layout.annotations:
        if ann.text:
            ann.yshift = (ann.yshift or 0) + SUBPLOT_TITLE_YSHIFT_PX


def style_fig(
    fig: go.Figure,
    title: str,
    *,
    subtitle: str | None = None,
    width: int = PLOT_W,
    height: int = PLOT_H,
) -> go.Figure:
    """Apply the locked layout to a figure. Returns the figure for chaining.

    Title is rendered bold via HTML; do not pre-wrap in `<b>...</b>`.
    `subtitle`, if given, renders below the title (not bold, 25% smaller)
    verbatim (no Title-Casing) so it can hold a free-form descriptive line.
    `width` and `height` are keyword-only. Pass them here rather than
    via `fig.update_layout(...)` after style_fig: the left-edge of the
    title and legend is computed as `MARGIN_L / width`, so a later width
    override would break alignment between title, legend, and plot area.
    """
    left_paper = MARGIN_L / width
    title_spec = dict(
        text = f"<b>{title_case(title)}</b>",
        font = dict(family=FONT_FAMILY, size=TITLE_SIZE, color=TEXT_COLOR),
        xref = "container",
        x = left_paper,
        xanchor = "left",
        yref = "container",
        y = 1 - TITLE_TOP_PAD_PX / height,
        yanchor = "top",
    )
    if subtitle:
        title_spec["subtitle"] = dict(
            text = subtitle,
            font = dict(family=FONT_FAMILY, size=SUBTITLE_SIZE, color=TEXT_COLOR),
        )
    fig.update_layout(
        title = title_spec,
        font = dict(family=FONT_FAMILY, size=12, color=TEXT_COLOR),
        paper_bgcolor = BG_COLOR,
        plot_bgcolor = BG_COLOR,
        width = width,
        height = height,
        xaxis = dict(
            title = dict(font=dict(size = 14)),
            tickfont = dict(size = 12),
            showgrid = False,
        ),
        yaxis = dict(
            title = dict(font=dict(size = 14)),
            tickfont = dict(size = 12),
            gridcolor = GRID_COLOR,
        ),
        legend = dict(
            orientation = "h",
            yanchor = "bottom",
            y = 1.01,
            xanchor = "left",
            x = 0,
            font = dict(size = 12),
        ),
        margin = dict(t = MARGIN_T, r = MARGIN_R, b = MARGIN_B, l = MARGIN_L),
    )
    _finalize_axes(fig)
    return fig
