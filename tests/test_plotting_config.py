"""Tests for carbon_forecast.plotting.config."""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

from carbon_forecast.plotting.config import (
    ENERGY_PALETTE,
    FONT_FAMILY,
    GRID_COLOR,
    MODEL_PALETTE,
    PLOT_H,
    PLOT_W,
    REGIONAL_PALETTE,
    apply_defaults,
    style_fig,
)


def test_regional_palette_has_five_zones():
    assert set(REGIONAL_PALETTE) == {"BE", "FI", "SG", "US-MIDA-PJM", "US-NY-NYIS"}


def test_energy_palette_has_carboncast_sources():
    must_include = {"gas", "coal", "oil", "nuclear", "wind", "solar", "hydro", "biomass"}
    assert must_include.issubset(set(ENERGY_PALETTE))


def test_model_palette_has_three_models():
    assert set(MODEL_PALETTE) == {"open", "em_operational", "carboncast_faithful"}


def test_apply_defaults_sets_plotly_white():
    pio.templates.default = "plotly"  # reset to a non-target value
    apply_defaults()
    assert pio.templates.default == "plotly_white"


def test_style_fig_wraps_title_in_bold():
    fig = go.Figure()
    style_fig(fig, "My title")
    assert fig.layout.title.text == "<b>My title</b>"


def test_style_fig_applies_locked_font_and_dimensions():
    fig = go.Figure()
    style_fig(fig, "x")
    assert fig.layout.font.family == FONT_FAMILY
    assert fig.layout.width == PLOT_W
    assert fig.layout.height == PLOT_H


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
