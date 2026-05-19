"""Tests for the extraction orchestration.

Mocks the clients at the object boundary; no HTTP. Uses tmp_path for the
data root so writes go to a real (temporary) filesystem.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from carbon_forecast.data.em_client import EMAPIError, SandboxModeError
from carbon_forecast.data.extract import (
    extract_em_history,
    extract_weather_history,
    iter_months,
    iter_sub_windows,
    month_window,
)
from carbon_forecast.data.weather_client import WeatherAPIError
from carbon_forecast.data.zones import Zone


@pytest.fixture
def be() -> Zone:
    return Zone("BE", 50.5, 4.5, "Belgium")


def _ci_payload(times: list[str]) -> dict:
    return {
        "zone": "BE",
        "history": [{"datetime": t, "carbonIntensity": 100 + i} for i, t in enumerate(times)],
    }


def _weather_payload(times: list[str]) -> dict:
    return {
        "hourly": {
            "time": times,
            "temperature_2m": [5.0] * len(times),
            "shortwave_radiation": [0.0] * len(times),
        }
    }


# --- iter_months / month_window ---------------------------------------------


def test_iter_months_inclusive():
    assert list(iter_months(date(2024, 1, 15), date(2024, 3, 1))) == [
        (2024, 1),
        (2024, 2),
        (2024, 3),
    ]


def test_iter_months_year_rollover():
    assert list(iter_months(date(2023, 11, 1), date(2024, 2, 1))) == [
        (2023, 11),
        (2023, 12),
        (2024, 1),
        (2024, 2),
    ]


def test_iter_months_rejects_end_before_start():
    with pytest.raises(ValueError, match=">="):
        list(iter_months(date(2024, 3, 1), date(2024, 1, 1)))


def test_month_window_january():
    s, e = month_window(2024, 1)
    assert s == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert e == datetime(2024, 2, 1, tzinfo=timezone.utc)


def test_month_window_december_rollover():
    s, e = month_window(2024, 12)
    assert s == datetime(2024, 12, 1, tzinfo=timezone.utc)
    assert e == datetime(2025, 1, 1, tzinfo=timezone.utc)


# --- iter_sub_windows -------------------------------------------------------


def test_iter_sub_windows_january_2024_yields_four():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 2, 1, tzinfo=timezone.utc)
    subs = list(iter_sub_windows(start, end, max_days=10))
    assert subs == [
        (start, datetime(2024, 1, 11, tzinfo=timezone.utc)),
        (datetime(2024, 1, 11, tzinfo=timezone.utc), datetime(2024, 1, 21, tzinfo=timezone.utc)),
        (datetime(2024, 1, 21, tzinfo=timezone.utc), datetime(2024, 1, 31, tzinfo=timezone.utc)),
        (datetime(2024, 1, 31, tzinfo=timezone.utc), end),
    ]
    # All sub-windows respect the 10-day cap.
    assert all((e - s).days <= 10 for s, e in subs)


def test_iter_sub_windows_short_interval_one_chunk():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 6, tzinfo=timezone.utc)
    assert list(iter_sub_windows(start, end, max_days=10)) == [(start, end)]


def test_iter_sub_windows_exact_max_days_one_chunk():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 11, tzinfo=timezone.utc)  # exactly 10 days
    assert list(iter_sub_windows(start, end, max_days=10)) == [(start, end)]


def test_iter_sub_windows_empty_interval():
    t = datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert list(iter_sub_windows(t, t)) == []


# --- extract_em_history ------------------------------------------------------


def test_em_extract_writes_and_skips(tmp_path: Path, be: Zone):
    # Each sub-window returns one canned record so we can verify concatenation.
    client = MagicMock()
    counter = {"i": 0}

    def fake_pull(zone, sub_start, sub_end):
        counter["i"] += 1
        return _ci_payload([sub_start.strftime("%Y-%m-%dT%H:%M:%SZ")])

    client.get_carbon_intensity_past_range.side_effect = fake_pull

    r1 = extract_em_history(
        client,
        zones=[be],
        endpoints=["carbon-intensity/past-range"],
        start=date(2024, 1, 1),
        end=date(2024, 2, 1),
        data_root=tmp_path,
    )
    assert r1.months_pulled == 2
    assert r1.months_skipped == 0
    assert r1.months_failed == []
    # January = 4 sub-windows, February (29 days, 2024 leap) = 3 sub-windows.
    assert client.get_carbon_intensity_past_range.call_count == 7
    # Total rows = 4 + 3 = 7 (one canned record per sub-window).
    assert r1.records_written == 7
    assert (tmp_path / "raw/em/BE/carbon-intensity/past-range/2024-01.parquet").exists()
    assert (tmp_path / "raw/em/BE/carbon-intensity/past-range/2024-02.parquet").exists()

    # Re-run with same args: no API calls, both months skipped.
    client.get_carbon_intensity_past_range.reset_mock(return_value=False, side_effect=False)
    r2 = extract_em_history(
        client,
        zones=[be],
        endpoints=["carbon-intensity/past-range"],
        start=date(2024, 1, 1),
        end=date(2024, 2, 1),
        data_root=tmp_path,
    )
    assert r2.months_pulled == 0
    assert r2.months_skipped == 2
    assert client.get_carbon_intensity_past_range.call_count == 0


def test_em_extract_continues_on_failure(tmp_path: Path, be: Zone):
    # Fail any sub-window whose start is in February; succeed otherwise.
    client = MagicMock()

    def fake_pull(zone, sub_start, sub_end):
        if sub_start.month == 2:
            raise EMAPIError("500 boom")
        return _ci_payload([sub_start.strftime("%Y-%m-%dT%H:%M:%SZ")])

    client.get_carbon_intensity_past_range.side_effect = fake_pull

    report = extract_em_history(
        client,
        zones=[be],
        endpoints=["carbon-intensity/past-range"],
        start=date(2024, 1, 1),
        end=date(2024, 3, 1),
        data_root=tmp_path,
    )
    assert report.months_pulled == 2  # Jan and Mar succeed; Feb fails on first sub-window.
    assert len(report.months_failed) == 1
    failed = report.months_failed[0]
    assert failed[:4] == ("BE", "carbon-intensity/past-range", 2024, 2)
    assert "EMAPIError" in failed[4]
    assert (tmp_path / "raw/em/BE/carbon-intensity/past-range/2024-01.parquet").exists()
    assert not (tmp_path / "raw/em/BE/carbon-intensity/past-range/2024-02.parquet").exists()
    assert (tmp_path / "raw/em/BE/carbon-intensity/past-range/2024-03.parquet").exists()


def test_em_extract_records_sandbox_as_failure(tmp_path: Path, be: Zone):
    client = MagicMock()
    client.get_carbon_intensity_past_range.side_effect = SandboxModeError("sandbox")

    report = extract_em_history(
        client,
        zones=[be],
        endpoints=["carbon-intensity/past-range"],
        start=date(2024, 1, 1),
        end=date(2024, 1, 1),
        data_root=tmp_path,
    )
    assert report.months_pulled == 0
    assert len(report.months_failed) == 1
    assert "SandboxModeError" in report.months_failed[0][4]


def test_em_extract_dry_run_makes_no_calls(tmp_path: Path, be: Zone):
    client = MagicMock()
    progress: list[str] = []

    report = extract_em_history(
        client,
        zones=[be],
        endpoints=["carbon-intensity/past-range"],
        start=date(2024, 1, 1),
        end=date(2024, 2, 1),
        data_root=tmp_path,
        dry_run=True,
        on_progress=progress.append,
    )
    assert report.months_pulled == 2
    assert client.get_carbon_intensity_past_range.call_count == 0
    assert not any(tmp_path.rglob("*.parquet"))
    assert all("would pull" in line for line in progress)


def test_em_extract_unknown_endpoint_raises(tmp_path: Path, be: Zone):
    client = MagicMock()
    with pytest.raises(ValueError, match="unknown EM endpoint"):
        extract_em_history(
            client,
            zones=[be],
            endpoints=["bogus/past-range"],
            start=date(2024, 1, 1),
            end=date(2024, 1, 1),
            data_root=tmp_path,
        )


def test_em_extract_subwindows_cover_month(tmp_path: Path, be: Zone):
    """Verify the month is pulled as multiple <=10-day sub-windows."""
    client = MagicMock()
    client.get_carbon_intensity_past_range.return_value = _ci_payload(
        ["2024-01-01T00:00:00Z"]
    )

    extract_em_history(
        client,
        zones=[be],
        endpoints=["carbon-intensity/past-range"],
        start=date(2024, 1, 1),
        end=date(2024, 1, 1),
        data_root=tmp_path,
    )
    # January 2024 -> 4 sub-windows.
    assert client.get_carbon_intensity_past_range.call_count == 4
    calls = [c.args for c in client.get_carbon_intensity_past_range.call_args_list]
    # First sub-window starts at month start; last sub-window ends at month end.
    assert calls[0][1] == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert calls[-1][2] == datetime(2024, 2, 1, tzinfo=timezone.utc)
    # No sub-window exceeds 10 days.
    for args in calls:
        assert (args[2] - args[1]).days <= 10


# --- extract_weather_history -------------------------------------------------


def test_weather_extract_writes_and_skips(tmp_path: Path, be: Zone):
    client = MagicMock()
    client.get_archive.return_value = _weather_payload(
        ["2024-01-01T00:00", "2024-01-01T01:00"]
    )

    r1 = extract_weather_history(
        client,
        zones=[be],
        start=date(2024, 1, 1),
        end=date(2024, 1, 1),
        data_root=tmp_path,
    )
    assert r1.months_pulled == 1
    assert (tmp_path / "raw/weather/BE/2024-01.parquet").exists()
    args, _ = client.get_archive.call_args
    assert args == (50.5, 4.5, date(2024, 1, 1), date(2024, 1, 31))

    # Idempotent re-run.
    client.get_archive.reset_mock()
    r2 = extract_weather_history(
        client,
        zones=[be],
        start=date(2024, 1, 1),
        end=date(2024, 1, 1),
        data_root=tmp_path,
    )
    assert r2.months_skipped == 1
    assert client.get_archive.call_count == 0


def test_weather_extract_continues_on_failure(tmp_path: Path, be: Zone):
    client = MagicMock()
    client.get_archive.side_effect = [
        _weather_payload(["2024-01-01T00:00"]),
        WeatherAPIError("503"),
    ]
    report = extract_weather_history(
        client,
        zones=[be],
        start=date(2024, 1, 1),
        end=date(2024, 2, 1),
        data_root=tmp_path,
    )
    assert report.months_pulled == 1
    assert len(report.months_failed) == 1
    assert report.months_failed[0][:4] == ("BE", "weather", 2024, 2)


def test_weather_extract_dry_run_makes_no_calls(tmp_path: Path, be: Zone):
    client = MagicMock()
    report = extract_weather_history(
        client,
        zones=[be],
        start=date(2024, 1, 1),
        end=date(2024, 2, 1),
        data_root=tmp_path,
        dry_run=True,
    )
    assert report.months_pulled == 2
    assert client.get_archive.call_count == 0
