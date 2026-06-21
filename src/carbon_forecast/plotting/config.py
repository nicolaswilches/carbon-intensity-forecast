"""
Project-wide plot configuration.

Locked design language (project_thesis.md, with style refinements logged in
memory/feedback_style.md):

- Template: plotly_white
- Font: Arial. Title 20pt bold, subplot titles 12pt, axis titles 14pt,
  ticks 12pt, legend 12pt.
- Grid: light gray horizontal only (rgba(0,0,0,0.1)). No vertical grid.
- Legend: horizontal, anchored to the top-left of the plot area.

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

# libraries and imports
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio


# Initial configuration
# dimensions
PLOT_W: int = 1000 # default width
PLOT_H: int = 500 # default height 

# colors
# Match the report body font (acmart sigconf uses Linux Libertine). The chain
# falls back to an available serif for static export if Libertine is not installed.
FONT_FAMILY: str = "Linux Libertine O, Libertinus Serif, Times New Roman, serif"
GRID_COLOR: str = "rgba(0,0,0,0.1)" # grid color
BG_COLOR: str = "#FFFFFF" # background color (white, matches the report page)
TEXT_COLOR: str = "#1F1F1F" # font color

# text sizes and spacing
TITLE_SIZE: int = 18  # title font size
TITLE_TOP_OFFSET: int = 45 # title offset from top border
SUBTITLE_SIZE: int = 15 # subtitle font size
SUBPLOT_TITLE_SIZE: int = 12 # subplot title font size
AXIS_TITLE_SIZE: int = 12 # x/y axis title font size
SUBPLOT_TITLE_YSHIFT_PX: int = 10 # subplot title shift in pixels
LINE_WIDTH: float = 1.2 # line width for line charts

# margins
MARGIN_L: int = 100
MARGIN_R: int = 80
MARGIN_T: int = 150 # top margin of the chart area
MARGIN_B: int = 80

# Report figure geometry. acmart sigconf: single column ~3.34in (~241pt),
# full text width ~7.02in (~506pt). Figures are sized so that, included at
# \columnwidth or \textwidth, line and tick text stay legible. Single-column
# figures go in `figure`; full-width go in `figure*`.
FIG_W_COLUMN: int = 470   # px; include at width=\columnwidth
FIG_W_FULL: int = 980     # px; include at width=\textwidth (figure*)
FIG_H_DEFAULT: int = 320  # px; default single-row height
# One font size for every report figure (axis titles, ticks, legend, base). The
# figures are scaled by ~0.68 on inclusion, so 13 px renders near the acmart 9 pt
# body size, keeping figure text unified with the report text.
REPORT_FONT: int = 13


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


def _lerp_hex(c1: str, c2: str, t: float) -> str:
    """Linear interpolation between two #RRGGBB colors at fraction t in [0, 1]."""
    a = tuple(int(c1.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    b = tuple(int(c2.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    r, g, bl = (round(a[j] + (b[j] - a[j]) * t) for j in range(3))
    return f"#{r:02X}{g:02X}{bl:02X}"


def discrete_percentile_colorscale(
    n_bins: int = 20,
    low_anchor: str | None = None,
    high: str = CI_COLOR,
) -> list[list]:
    """Build a discrete plotly colorscale of `n_bins` equal bands.

    Each band is a flat color, ramping from `low_anchor` (a hue just above the
    canvas background by default) to `high` (the carbon-intensity orange). Used
    on a 0..100 percentile axis so each band spans 100/n_bins percentile points
    (20 bins -> one hue per 5 percentile). The lowest band sits just above
    BG_COLOR; the highest is exactly `high`.
    """
    if low_anchor is None:
        low_anchor = _lerp_hex(BG_COLOR, high, 0.08)
    colors = [_lerp_hex(low_anchor, high, k / (n_bins - 1)) for k in range(n_bins)]
    scale: list[list] = []
    for k in range(n_bins):
        scale.append([k / n_bins, colors[k]])
        scale.append([(k + 1) / n_bins, colors[k]])
    return scale


def _title_case_word(w: str) -> str:
    """
    Title-case one whitespace-delimited token, preserving acronyms and units.

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
    """
    Style and retitle every axis, and Title-Case every trace name, in place.

    Iterates all axes (xaxis, xaxis2, ... yaxis, yaxis2, ...) so multi-row
    subplots get the same tick font, grid policy, and Title-Cased titles as
    a single-plot figure, not just the first subplot.
    """

    for key in list(fig.layout):
        if key.startswith("xaxis"):
            ax = fig.layout[key]
            ax.update(tickfont=dict(size=12), showgrid=False)
            if ax.title is not None and ax.title.text:
                ax.title.update(text=title_case(ax.title.text), font=dict(size=AXIS_TITLE_SIZE))
        elif key.startswith("yaxis"):
            ax = fig.layout[key]
            ax.update(tickfont=dict(size=12), gridcolor=GRID_COLOR)
            if ax.title is not None and ax.title.text:
                ax.title.update(text=title_case(ax.title.text), font=dict(size=AXIS_TITLE_SIZE))

    for trace in fig.data:
        name = getattr(trace, "name", None)
        if isinstance(name, str) and name:
            trace.name = title_case(name.replace("_", " "))
        # Thicken thin line swatches so the legend color is visible. Only
        # Scatter traces carry a top-level `.line`; Bar/Heatmap do not.
        if isinstance(trace, go.Scatter) and (trace.mode is None or "lines" in trace.mode):
            if trace.line.width is None:
                trace.line.width = LINE_WIDTH
    # Lift subplot titles (make_subplots / facet labels are annotations) off
    # the top of their panels so they do not crowd the chart.

    for ann in fig.layout.annotations:
        if ann.text:
            ann.font.update(
                family=FONT_FAMILY,
                size=SUBPLOT_TITLE_SIZE,
                color=TEXT_COLOR,
            )
            ann.yshift = (ann.yshift or 0) + SUBPLOT_TITLE_YSHIFT_PX


def style_fig(
    fig: go.Figure,
    title: str,
    *,
    subtitle: str | None = None,
    width: int = PLOT_W,
    height: int = PLOT_H,
) -> go.Figure:
    """
    Apply the locked layout to a figure. Returns the figure for chaining.

    - Title is rendered bold via HTML; do not pre-wrap in `<b>...</b>`.
    - `subtitle`, if given, renders below the title.
    - verbatim (no Title-Casing) so it can hold a free-form descriptive line.
    - `width` and `height` are keyword-only. Pass them here rather than
      via `fig.update_layout(...)` after style_fig: the left-edge of the
      title and legend is computed as `MARGIN_L / width`, so a later width
      override would break alignment between title, legend, and plot area.
    """

    left_paper = MARGIN_L / width

    title_spec = dict(
        text = f"<b>{title_case(title)}</b>",
        font = dict(family = FONT_FAMILY, size = TITLE_SIZE, color = TEXT_COLOR),
        xref = "container",
        x = left_paper,
        xanchor = "left",
        yref = "container",
        y = 1 - TITLE_TOP_OFFSET / height,
        yanchor = "top",
    )

    if subtitle:
        title_spec["text"] = (
            f"<span style='line-height: 1.0;'><b>{title_case(title)}</b></span><br>"
            f"<span style='font-family: {FONT_FAMILY}; font-size: {SUBTITLE_SIZE}px; color: {TEXT_COLOR}; font-weight: normal;'>{subtitle}</span>"
        )

    subtitle_shift = 15 if subtitle else 0
    legend_top_offset = subtitle_shift + 70
    margin_t = MARGIN_T + subtitle_shift

    fig.update_layout(
        title=title_spec,
        font=dict(family=FONT_FAMILY, size=12, color=TEXT_COLOR),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        width=width,
        height=height,
        xaxis=dict(
            title=dict(font=dict(size=AXIS_TITLE_SIZE)),
            tickfont=dict(size=10),
            showgrid=False,
        ),
        yaxis=dict(
            title=dict(font=dict(size=AXIS_TITLE_SIZE)),
            tickfont=dict(size=12),
            gridcolor=GRID_COLOR,
        ),
        legend=dict(
            orientation="h",
            xref="container",
            yref="container",
            xanchor="left",
            yanchor="top",
            x = left_paper,
            y = 1 - legend_top_offset / height,
            font=dict(size=12),
        ),
        margin=dict(t=margin_t, r=MARGIN_R, b=MARGIN_B, l=MARGIN_L),
    )
    _finalize_axes(fig)
    return fig


def style_report_fig(
    fig: go.Figure,
    *,
    span: str = "column",
    height: int | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    legend: bool = True,
) -> go.Figure:
    """Lean styling for figures embedded in the report.

    No in-figure title (the LaTeX \\caption carries it). White background, report
    serif font, compact margins, and a width tied to the acmart geometry:
    span="column" fits \\columnwidth, span="full" fits \\textwidth. Pass `height`
    for tall small-multiples. Export to PDF for the report.
    """
    width = FIG_W_FULL if span == "full" else FIG_W_COLUMN
    height = height or FIG_H_DEFAULT
    fig.update_layout(
        title=None,
        font=dict(family=FONT_FAMILY, size=REPORT_FONT, color=TEXT_COLOR),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        width=width,
        height=height,
        margin=dict(t=(36 if legend else 14), r=18, b=18, l=60),
        showlegend=legend,
    )
    if legend:
        fig.update_layout(legend=dict(
            orientation="h", x=0, xanchor="left", y=1.02, yanchor="bottom",
            font=dict(size=REPORT_FONT),
        ))
    # automargin grows the bottom only as far as the x labels/title need, so the
    # small base margin removes the dead space below the plot without clipping.
    fig.update_xaxes(showgrid=False, automargin=True, tickfont=dict(size=REPORT_FONT),
                     title=dict(text=xlabel, font=dict(size=REPORT_FONT)))
    fig.update_yaxes(gridcolor=GRID_COLOR, tickfont=dict(size=REPORT_FONT),
                     title=dict(text=ylabel, font=dict(size=REPORT_FONT)))
    return fig
