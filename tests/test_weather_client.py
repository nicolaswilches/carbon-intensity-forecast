"""Tests for WeatherClient.

Same pattern as the EMClient tests: mock requests.Session, assert on
URL/params/retry behavior. No live HTTP.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest
import requests

from carbon_forecast.data.weather_client import (
    DEFAULT_ARCHIVE_BASE_URL,
    DEFAULT_FORECAST_BASE_URL,
    HOURLY_VARIABLES,
    WeatherAPIError,
    WeatherClient,
)


def _fake_response(status: int, json_payload: dict | None = None, text: str = "") -> MagicMock:
    response = MagicMock(spec=requests.Response)
    response.status_code = status
    response.json.return_value = json_payload or {}
    response.text = text
    return response


def _client(session: MagicMock, *, backoff_base: float = 0.0, max_retries: int = 3) -> WeatherClient:
    return WeatherClient(
        session=session,
        backoff_base=backoff_base,
        max_retries=max_retries,
    )


# --- coord and window validation ---------------------------------------------


def test_invalid_latitude_raises():
    client = _client(MagicMock())
    with pytest.raises(ValueError, match="latitude"):
        client.get_archive(95.0, 0.0, date(2024, 1, 1), date(2024, 1, 2))


def test_invalid_longitude_raises():
    client = _client(MagicMock())
    with pytest.raises(ValueError, match="longitude"):
        client.get_archive(0.0, 200.0, date(2024, 1, 1), date(2024, 1, 2))


def test_start_after_end_raises():
    client = _client(MagicMock())
    with pytest.raises(ValueError, match="<= end"):
        client.get_archive(50.0, 4.0, date(2024, 2, 1), date(2024, 1, 1))


def test_forecast_days_out_of_range_raises():
    client = _client(MagicMock())
    with pytest.raises(ValueError, match="forecast_days"):
        client.get_gfs_forecast(50.0, 4.0, forecast_days=0)
    with pytest.raises(ValueError, match="forecast_days"):
        client.get_gfs_forecast(50.0, 4.0, forecast_days=17)


def test_negative_past_days_raises():
    client = _client(MagicMock())
    with pytest.raises(ValueError, match="past_days"):
        client.get_gfs_forecast(50.0, 4.0, past_days=-1)


# --- archive happy path ------------------------------------------------------


def test_archive_request_shape():
    payload = {"hourly": {"time": [], "temperature_2m": []}}
    session = MagicMock()
    session.get.return_value = _fake_response(200, payload)
    client = _client(session)

    result = client.get_archive(50.85, 4.35, date(2024, 1, 1), date(2024, 1, 31))

    assert result == payload
    args, kwargs = session.get.call_args
    assert args[0] == DEFAULT_ARCHIVE_BASE_URL
    assert kwargs["params"] == {
        "latitude": "50.85",
        "longitude": "4.35",
        "start_date": "2024-01-01",
        "end_date": "2024-01-31",
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "UTC",
    }


def test_custom_variables_override_default():
    session = MagicMock()
    session.get.return_value = _fake_response(200, {"hourly": {}})
    client = _client(session)
    client.get_archive(
        50.0,
        4.0,
        date(2024, 1, 1),
        date(2024, 1, 2),
        variables=["temperature_2m", "precipitation"],
    )
    _, kwargs = session.get.call_args
    assert kwargs["params"]["hourly"] == "temperature_2m,precipitation"


# --- forecast happy path -----------------------------------------------------


def test_gfs_request_shape():
    payload = {"hourly": {"time": [], "temperature_2m": []}}
    session = MagicMock()
    session.get.return_value = _fake_response(200, payload)
    client = _client(session)

    client.get_gfs_forecast(50.85, 4.35, forecast_days=4, past_days=1)

    args, kwargs = session.get.call_args
    assert args[0] == DEFAULT_FORECAST_BASE_URL
    assert kwargs["params"] == {
        "latitude": "50.85",
        "longitude": "4.35",
        "forecast_days": "4",
        "past_days": "1",
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "UTC",
    }


# --- retry behavior ----------------------------------------------------------


def test_retries_on_429_then_succeeds():
    session = MagicMock()
    session.get.side_effect = [
        _fake_response(429, text="rate limited"),
        _fake_response(200, {"hourly": {}}),
    ]
    client = _client(session)
    client.get_archive(50.0, 4.0, date(2024, 1, 1), date(2024, 1, 2))
    assert session.get.call_count == 2


def test_retries_on_5xx_then_succeeds():
    session = MagicMock()
    session.get.side_effect = [
        _fake_response(502, text="bad gateway"),
        _fake_response(200, {"hourly": {}}),
    ]
    client = _client(session)
    client.get_archive(50.0, 4.0, date(2024, 1, 1), date(2024, 1, 2))
    assert session.get.call_count == 2


def test_retries_on_connection_error_then_succeeds():
    session = MagicMock()
    session.get.side_effect = [
        requests.ConnectionError("network down"),
        _fake_response(200, {"hourly": {}}),
    ]
    client = _client(session)
    client.get_archive(50.0, 4.0, date(2024, 1, 1), date(2024, 1, 2))
    assert session.get.call_count == 2


def test_retry_exhaustion_raises():
    session = MagicMock()
    session.get.return_value = _fake_response(500, text="boom")
    client = _client(session, max_retries=2)
    with pytest.raises(WeatherAPIError, match="500 after 2 retries"):
        client.get_archive(50.0, 4.0, date(2024, 1, 1), date(2024, 1, 2))
    assert session.get.call_count == 3


def test_400_does_not_retry():
    session = MagicMock()
    session.get.return_value = _fake_response(400, text="bad request")
    client = _client(session)
    with pytest.raises(WeatherAPIError, match="400"):
        client.get_archive(50.0, 4.0, date(2024, 1, 1), date(2024, 1, 2))
    assert session.get.call_count == 1
