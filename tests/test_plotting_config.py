"""Tests for carbon_forecast.plotting.config."""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

from carbon_forecast.plotting.config import (
    ENERGY_PALETTE,
    ENERGY_SOURCE_ORDER,
    FONT_FAMILY,
    GRID_COLOR,
    MODEL_PALETTE,
    PLOT_H,
    PLOT_W,
    REGIONAL_PALETTE,
    apply_defaults,
    style_fig,
    title_case,
)


def test_regional_palette_has_five_zones():
    assert set(REGIONAL_PALETTE) == {"BE", "FI", "SG", "US-MIDA-PJM", "US-NY-NYIS"}


def test_energy_palette_has_carboncast_sources():
    must_include = {"gas", "coal", "oil", "nuclear", "wind", "solar", "hydro", "biomass"}
    assert must_include.issubset(set(ENERGY_PALETTE))


def test_energy_source_order_matches_palette_keys():
    # ENERGY_SOURCE_ORDER is the canonical display order; every entry must have a color
    # in ENERGY_PALETTE, and vice versa.
    assert set(ENERGY_SOURCE_ORDER) == set(ENERGY_PALETTE)


def test_energy_source_order_starts_with_nuclear_baseload():
    # User-chosen convention: nuclear sits at the bottom of stacked-area charts.
    assert ENERGY_SOURCE_ORDER[0] == "nuclear"


def test_energy_source_order_ends_with_unknown():
    # Unclassified energy sits at the very top of stacks.
    assert ENERGY_SOURCE_ORDER[-1] == "unknown"


def test_hydro_discharge_listed_before_hydro():
    # User convention: discharge sits below hydro proper in the stack.
    idx_disc = ENERGY_SOURCE_ORDER.index("hydro_discharge")
    idx_hydro = ENERGY_SOURCE_ORDER.index("hydro")
    assert idx_disc < idx_hydro


# --- title_case helper ------------------------------------------------------


def test_title_case_simple_lowercase():
    assert title_case("hydro discharge") == "Hydro Discharge"


def test_title_case_preserves_acronyms_and_units():
    assert title_case("Hour (UTC)") == "Hour (UTC)"
    assert title_case("Mean MW") == "Mean MW"
    assert title_case("Annual TWh") == "Annual TWh"
    assert title_case("gCO2eq/kWh stays mixed") == "gCO2eq/kWh Stays Mixed"


def test_title_case_handles_hyphenated_words():
    assert title_case("cross-border flows") == "Cross-Border Flows"


def test_title_case_handles_parenthesised_lowercase():
    assert title_case("production (hourly, mw)") == "Production (Hourly, Mw)"


def test_title_case_leaves_numbers_alone():
    assert title_case("January 2024 summary") == "January 2024 Summary"


# --- style_fig auto-title-cases ---------------------------------------------


def test_style_fig_title_cased_automatically():
    fig = go.Figure()
    style_fig(fig, "annual gross flows per zone")
    assert fig.layout.title.text == "<b>Annual Gross Flows Per Zone</b>"


def test_style_fig_retitles_axis_labels_and_trace_names():
    fig = go.Figure()
    fig.update_xaxes(title_text="hour of day (utc)")
    fig.update_yaxes(title_text="mean carbon intensity")
    fig.add_trace(go.Scatter(x=[1, 2], y=[3, 4], name="hydro_discharge"))
    style_fig(fig, "x")
    assert fig.layout.xaxis.title.text == "Hour Of Day (Utc)"
    assert fig.layout.yaxis.title.text == "Mean Carbon Intensity"
    assert fig.data[0].name == "Hydro Discharge"


def test_style_fig_styles_every_subplot_axis():
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=3, cols=1)
    fig.update_yaxes(title_text="mw", row=2, col=1)
    style_fig(fig, "x")
    # Grid color and tick font reach axes beyond the first subplot.
    assert fig.layout.yaxis2.gridcolor == GRID_COLOR
    assert fig.layout.yaxis3.gridcolor == GRID_COLOR
    assert fig.layout.xaxis2.showgrid is False
    # Titles on non-first axes are still Title-Cased.
    assert fig.layout.yaxis2.title.text == "Mw"


def test_model_palette_has_three_models():
    assert set(MODEL_PALETTE) == {"open", "em_operational", "carboncast_faithful"}


def test_apply_defaults_sets_plotly_white():
    pio.templates.default = "plotly"  # reset to a non-target value
    apply_defaults()
    assert pio.templates.default == "plotly_white"


def test_style_fig_wraps_title_in_bold():
    fig = go.Figure()
    style_fig(fig, "My title")
    # Title is also Title-Cased automatically.
    assert fig.layout.title.text == "<b>My Title</b>"


def test_style_fig_applies_locked_font_and_dimensions():
    fig = go.Figure()
    style_fig(fig, "x")
    assert fig.layout.font.family == FONT_FAMILY
    assert fig.layout.width == PLOT_W
    assert fig.layout.height == PLOT_H


def test_style_fig_respects_custom_width_and_height():
    from carbon_forecast.plotting.config import MARGIN_L

    fig = go.Figure()
    style_fig(fig, "x", width=1800, height=600)
    assert fig.layout.width == 1800
    assert fig.layout.height == 600
    # Critical: title.x and legend.x derive from the *actual* width so the
    # left edges of title, legend, and plot area stay aligned at any size.
    expected_left = MARGIN_L / 1800
    assert fig.layout.title.x == expected_left
    assert fig.layout.legend.x == expected_left


def test_style_fig_uses_grid_color_on_y_axis():
    fig = go.Figure()
    style_fig(fig, "x")
    assert fig.layout.yaxis.gridcolor == GRID_COLOR


def test_style_fig_legend_aligned_with_plot_area():
    from carbon_forecast.plotting.config import PLOT_AREA_LEFT_PAPER

    fig = go.Figure()
    style_fig(fig, "x")
    legend = fig.layout.legend
    assert legend.orientation == "h"
    assert legend.xanchor == "left"
    # Legend left edge sits on the plot area left edge (same paper-x as title).
    assert legend.x == PLOT_AREA_LEFT_PAPER
    # Legend sits just above the plot area, well below the title row.
    assert 1.0 < legend.y < 1.1


def test_style_fig_title_aligned_with_plot_area():
    from carbon_forecast.plotting.config import PLOT_AREA_LEFT_PAPER

    fig = go.Figure()
    style_fig(fig, "x")
    title = fig.layout.title
    # Title left edge sits on the plot area left edge.
    assert title.xanchor == "left"
    assert title.x == PLOT_AREA_LEFT_PAPER
    # Title anchored to the top of the figure with breathing room above.
    assert title.yanchor == "top"
    assert title.y is not None and title.y >= 0.85


def test_style_fig_title_and_legend_do_not_overlap():
    """Title bottom must sit above legend top in paper coords."""
    from carbon_forecast.plotting.config import PLOT_H, MARGIN_T

    fig = go.Figure()
    style_fig(fig, "x")
    plot_top_paper = 1 - MARGIN_T / PLOT_H
    plot_height_paper = plot_top_paper - (fig.layout.margin.b / PLOT_H)

    # Title bottom in paper coords (approx: title.y - font_size / PLOT_H).
    title_font_paper = fig.layout.title.font.size / PLOT_H
    title_bottom_paper = fig.layout.title.y - title_font_paper

    # Legend top in paper coords (approx).
    legend_bottom_paper = plot_top_paper + (fig.layout.legend.y - 1.0) * plot_height_paper
    legend_font_paper = fig.layout.legend.font.size / PLOT_H
    legend_top_paper = legend_bottom_paper + legend_font_paper

    # Title's bottom must clear legend's top by at least 4% of figure height.
    assert title_bottom_paper - legend_top_paper > 0.04


def test_style_fig_top_margin_leaves_room_for_title_and_legend():
    fig = go.Figure()
    style_fig(fig, "x")
    # Top margin holds the title plus the legend strip above the plot area.
    assert fig.layout.margin.t >= 120


def test_style_fig_returns_same_figure():
    fig = go.Figure()
    out = style_fig(fig, "x")
    assert out is fig
