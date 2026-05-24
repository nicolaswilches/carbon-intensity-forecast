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


def test_style_fig_legend_anchored_top_left():
    fig = go.Figure()
    style_fig(fig, "x")
    legend = fig.layout.legend
    assert legend.orientation == "h"
    assert legend.xanchor == "left"
    assert legend.x == 0


def test_style_fig_returns_same_figure():
    fig = go.Figure()
    out = style_fig(fig, "x")
    assert out is fig
