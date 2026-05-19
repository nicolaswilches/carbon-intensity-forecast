"""Tests for the storage layer.

No live HTTP, no external state. Uses tmp_path for filesystem tests.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from carbon_forecast.data.storage import (
    em_forecast_path,
    em_history_path,
    flatten_em_carbon_intensity,
    flatten_em_power_breakdown,
    flatten_em_power_flows,
    flatten_weather_hourly,
    list_months_present,
    processed_path,
    read_parquet,
    weather_archive_path,
    write_parquet_atomic,
)


# --- path helpers ------------------------------------------------------------


def test_em_history_path():
    root = Path("/tmp/data")
    p = em_history_path("BE", "carbon-intensity/past-range", 2024, 3, root)
    assert p == root / "raw/em/BE/carbon-intensity/past-range/2024-03.parquet"


def test_weather_archive_path():
    root = Path("/tmp/data")
    p = weather_archive_path("FI", 2024, 12, root)
    assert p == root / "raw/weather/FI/2024-12.parquet"


def test_em_forecast_path():
    snap = datetime(2026, 6, 8, 12, 30, 0, tzinfo=timezone.utc)
    p = em_forecast_path("US-NY-NYIS", snap, Path("/tmp/data"))
    assert p == Path("/tmp/data/raw/em/forecasts/US-NY-NYIS/20260608T123000Z.parquet")


def test_em_forecast_path_rejects_naive():
    with pytest.raises(ValueError, match="timezone-aware"):
        em_forecast_path("BE", datetime(2026, 6, 8, 12, 0), Path("/tmp/data"))


def test_processed_path():
    assert processed_path("SG", Path("/tmp/data")) == Path("/tmp/data/processed/SG.parquet")


# --- flatten: carbon intensity ----------------------------------------------


def test_flatten_carbon_intensity_basic():
    payload = {
        "zone": "BE",
        "history": [
            {"datetime": "2024-01-01T00:00:00Z", "carbonIntensity": 120.5},
            {"datetime": "2024-01-01T01:00:00Z", "carbonIntensity": 130.0},
        ],
    }
    df = flatten_em_carbon_intensity(payload)
    assert list(df.columns) == ["carbon_intensity_gco2eq_kwh"]
    assert df.index.name == "datetime"
    assert str(df.index.tz) == "UTC"
    assert df["carbon_intensity_gco2eq_kwh"].tolist() == [120.5, 130.0]


def test_flatten_carbon_intensity_empty():
    df = flatten_em_carbon_intensity({"zone": "BE", "history": []})
    assert df.empty
    assert df.index.name == "datetime"


def test_flatten_carbon_intensity_dedupes_and_sorts():
    payload = {
        "history": [
            {"datetime": "2024-01-01T01:00:00Z", "carbonIntensity": 100},
            {"datetime": "2024-01-01T00:00:00Z", "carbonIntensity": 90},
            {"datetime": "2024-01-01T01:00:00Z", "carbonIntensity": 110},  # dup, keep last
        ]
    }
    df = flatten_em_carbon_intensity(payload)
    assert len(df) == 2
    assert df["carbon_intensity_gco2eq_kwh"].tolist() == [90, 110]


# --- flatten: power breakdown ------------------------------------------------


def test_flatten_power_breakdown():
    payload = {
        "history": [
            {
                "datetime": "2024-01-01T00:00:00Z",
                "powerProductionBreakdown": {"gas": 100, "coal": 50, "solar": None},
                "powerConsumptionBreakdown": {"gas": 80},
                "powerImportBreakdown": {"FR": 25, "US-MIDA-PJM": 10},  # intentionally ignored
                "powerExportBreakdown": {"DE": 5},                       # intentionally ignored
                "powerImportTotal": 35,
                "powerExportTotal": 5,
            }
        ]
    }
    df = flatten_em_power_breakdown(payload)
    row = df.iloc[0]
    assert row["prod_gas_mw"] == 100
    assert row["prod_coal_mw"] == 50
    assert "prod_solar_mw" not in df.columns  # None values dropped
    assert row["cons_gas_mw"] == 80
    assert row["import_total_mw"] == 35
    assert row["export_total_mw"] == 5
    # Partner-aggregated breakdown columns must NOT appear; that's power-flows territory.
    assert not any(c.startswith("import_") and c != "import_total_mw" for c in df.columns)
    assert not any(c.startswith("export_") and c != "export_total_mw" for c in df.columns)


def test_flatten_power_breakdown_empty():
    assert flatten_em_power_breakdown({"history": []}).empty


# --- flatten: power flows ----------------------------------------------------


def test_flatten_power_flows():
    payload = {
        "history": [
            {
                "datetime": "2024-01-01T00:00:00Z",
                "powerImports": {"FR": 25, "DE": 10},
                "powerExports": {"NL": 5},
            }
        ]
    }
    df = flatten_em_power_flows(payload)
    row = df.iloc[0]
    assert row["import_fr_mw"] == 25
    assert row["import_de_mw"] == 10
    assert row["export_nl_mw"] == 5


def test_flatten_power_flows_empty():
    assert flatten_em_power_flows({"history": []}).empty


# --- flatten: weather --------------------------------------------------------


def test_flatten_weather_hourly():
    payload = {
        "hourly": {
            "time": ["2024-01-01T00:00", "2024-01-01T01:00"],
            "temperature_2m": [5.2, 5.5],
            "shortwave_radiation": [0.0, 0.0],
        },
        "hourly_units": {"temperature_2m": "°C"},
    }
    df = flatten_weather_hourly(payload)
    assert list(df.columns) == ["temperature_2m", "shortwave_radiation"]
    assert df["temperature_2m"].tolist() == [5.2, 5.5]
    assert str(df.index.tz) == "UTC"


def test_flatten_weather_empty():
    assert flatten_weather_hourly({"hourly": {"time": []}}).empty


# --- write / read round-trip -------------------------------------------------


def test_write_parquet_atomic_round_trip(tmp_path: Path):
    df = pd.DataFrame(
        {"carbon_intensity_gco2eq_kwh": [120.0, 130.0]},
        index=pd.DatetimeIndex(
            ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"],
            tz="UTC",
            name="datetime",
        ),
    )
    path = tmp_path / "raw/em/BE/carbon-intensity/past-range/2024-01.parquet"
    written = write_parquet_atomic(df, path)
    assert written == path
    assert path.exists()

    df2 = read_parquet(path)
    pd.testing.assert_frame_equal(df, df2)


def test_write_parquet_atomic_overwrites(tmp_path: Path):
    path = tmp_path / "x.parquet"
    df1 = pd.DataFrame(
        {"x": [1]},
        index=pd.DatetimeIndex(["2024-01-01T00:00:00Z"], tz="UTC", name="datetime"),
    )
    df2 = pd.DataFrame(
        {"x": [2]},
        index=pd.DatetimeIndex(["2024-01-01T00:00:00Z"], tz="UTC", name="datetime"),
    )
    write_parquet_atomic(df1, path)
    write_parquet_atomic(df2, path)
    assert read_parquet(path)["x"].iloc[0] == 2


def test_write_parquet_leaves_no_temp_files(tmp_path: Path):
    path = tmp_path / "x.parquet"
    df = pd.DataFrame(
        {"x": [1]},
        index=pd.DatetimeIndex(["2024-01-01T00:00:00Z"], tz="UTC", name="datetime"),
    )
    write_parquet_atomic(df, path)
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".") and ".tmp" in p.name]
    assert leftovers == []


# --- list_months_present ----------------------------------------------------


def test_list_months_present(tmp_path: Path):
    df = pd.DataFrame(
        {"x": [1]},
        index=pd.DatetimeIndex(["2024-01-01T00:00:00Z"], tz="UTC", name="datetime"),
    )
    for ym in ("2024-01", "2024-02", "2024-12"):
        write_parquet_atomic(df, em_history_path("BE", "carbon-intensity/past-range", int(ym[:4]), int(ym[5:]), tmp_path))
    # noise files that should be ignored
    noise_dir = tmp_path / "raw/em/BE/carbon-intensity/past-range"
    (noise_dir / "README.md").write_text("ignore me")
    (noise_dir / "2024.parquet").write_text("wrong format")

    months = list_months_present("BE", "carbon-intensity/past-range", tmp_path)
    assert months == {(2024, 1), (2024, 2), (2024, 12)}


def test_list_months_present_missing_dir(tmp_path: Path):
    assert list_months_present("BE", "carbon-intensity/past-range", tmp_path) == set()
