"""Tests for carbon_forecast.evaluation.metrics."""

import numpy as np

from carbon_forecast.evaluation import mae, mape_pct, per_horizon, rmse, summary


def test_perfect_forecast_is_zero_error():
    y = np.array([[100.0, 200.0], [50.0, 75.0]])
    assert mae(y, y) == 0.0
    assert rmse(y, y) == 0.0
    assert mape_pct(y, y) == 0.0


def test_known_values():
    y = np.array([[100.0]])
    p = np.array([[110.0]])
    assert abs(mae(p, y) - 10.0) < 1e-9
    assert abs(rmse(p, y) - 10.0) < 1e-9
    assert abs(mape_pct(p, y) - 10.0) < 1e-9


def test_summary_has_all_keys_and_count():
    y = np.array([[1.0, 2.0], [3.0, 4.0]])
    s = summary(y + 1.0, y)
    assert set(s) == {"mape_pct", "mae", "rmse", "n_samples"}
    assert s["n_samples"] == 2


def test_per_horizon_returns_one_value_per_step():
    y = np.ones((5, 3))
    p = np.ones((5, 3)) * 2.0
    assert per_horizon(p, y, "mae").shape == (3,)
    assert np.allclose(per_horizon(p, y, "mae"), 1.0)


def test_nans_are_ignored():
    y = np.array([[100.0, np.nan]])
    p = np.array([[110.0, 999.0]])
    assert abs(mae(p, y) - 10.0) < 1e-9
    assert abs(mape_pct(p, y) - 10.0) < 1e-9


def test_unknown_metric_raises():
    y = np.ones((2, 2))
    try:
        per_horizon(y, y, "bogus")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown metric")
