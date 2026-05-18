"""Tests for EMClient.

All HTTP traffic is mocked at the requests.Session boundary so we never
hit the real API. Tests focus on the contract: auth header, URL/params,
sandbox detection, retry behavior, error mapping.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
import requests

from carbon_forecast.data.em_client import (
    DEFAULT_BASE_URL,
    EMAPIError,
    EMClient,
    SandboxModeError,
    _to_iso_z,
)


def _fake_response(status: int, json_payload: dict | None = None, text: str = "") -> MagicMock:
    response = MagicMock(spec=requests.Response)
    response.status_code = status
    response.json.return_value = json_payload or {}
    response.text = text
    return response


def _client(session: MagicMock, *, backoff_base: float = 0.0, max_retries: int = 3) -> EMClient:
    return EMClient(
        api_key="test-key",
        base_url=DEFAULT_BASE_URL,
        session=session,
        backoff_base=backoff_base,
        max_retries=max_retries,
    )


# --- construction ------------------------------------------------------------


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("EM_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="EM_API_KEY"):
        EMClient()


def test_explicit_key_overrides_env(monkeypatch):
    monkeypatch.setenv("EM_API_KEY", "env-key")
    client = EMClient(api_key="explicit")
    assert client._api_key == "explicit"


# --- validation --------------------------------------------------------------


def test_naive_datetime_raises():
    session = MagicMock()
    client = _client(session)
    naive = datetime(2024, 1, 1)
    aware = datetime(2024, 1, 2, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="timezone-aware"):
        client.get_carbon_intensity_past_range("BE", naive, aware)


def test_start_not_before_end_raises():
    session = MagicMock()
    client = _client(session)
    t = datetime(2024, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="strictly before"):
        client.get_carbon_intensity_past_range("BE", t, t)


def test_iso_z_formatting():
    dt = datetime(2024, 1, 15, 6, 30, 45, 123456, tzinfo=timezone.utc)
    assert _to_iso_z(dt) == "2024-01-15T06:30:45Z"


# --- happy path --------------------------------------------------------------


def test_request_shape_and_auth_header():
    payload = {"history": [{"carbonIntensity": 120}], "zone": "BE"}
    session = MagicMock()
    session.get.return_value = _fake_response(200, payload)
    client = _client(session)

    result = client.get_carbon_intensity_past_range(
        "BE",
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 2, 1, tzinfo=timezone.utc),
    )

    assert result == payload
    session.get.assert_called_once()
    args, kwargs = session.get.call_args
    assert args[0] == f"{DEFAULT_BASE_URL}/carbon-intensity/past-range"
    assert kwargs["headers"] == {"auth-token": "test-key"}
    assert kwargs["params"] == {
        "zone": "BE",
        "start": "2024-01-01T00:00:00Z",
        "end": "2024-02-01T00:00:00Z",
    }


def test_each_endpoint_hits_correct_path():
    session = MagicMock()
    session.get.return_value = _fake_response(200, {"history": []})
    client = _client(session)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    end = datetime(2024, 1, 2, tzinfo=timezone.utc)

    client.get_carbon_intensity_past_range("BE", start, end)
    client.get_power_breakdown_past_range("BE", start, end)
    client.get_power_flows_past_range("BE", start, end)

    called_urls = [call.args[0] for call in session.get.call_args_list]
    assert called_urls == [
        f"{DEFAULT_BASE_URL}/carbon-intensity/past-range",
        f"{DEFAULT_BASE_URL}/power-breakdown/past-range",
        f"{DEFAULT_BASE_URL}/power-flows/past-range",
    ]


# --- sandbox detection -------------------------------------------------------


def test_sandbox_disclaimer_raises():
    payload = {
        "_disclaimer": "SANDBOX MODE - this dataset is synthetic.",
        "history": [{"carbonIntensity": 999}],
    }
    session = MagicMock()
    session.get.return_value = _fake_response(200, payload)
    client = _client(session)
    with pytest.raises(SandboxModeError, match="sandbox disclaimer"):
        client.get_carbon_intensity_past_range(
            "BE",
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 2, tzinfo=timezone.utc),
        )


def test_sandbox_estimation_method_raises():
    payload = {
        "history": [
            {"carbonIntensity": 120, "estimationMethod": "MEASURED"},
            {"carbonIntensity": 999, "estimationMethod": "SANDBOX_MODE_DATA"},
        ]
    }
    session = MagicMock()
    session.get.return_value = _fake_response(200, payload)
    client = _client(session)
    with pytest.raises(SandboxModeError, match="SANDBOX_MODE_DATA"):
        client.get_carbon_intensity_past_range(
            "BE",
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 2, tzinfo=timezone.utc),
        )


def test_non_sandbox_disclaimer_passes_through():
    payload = {
        "_disclaimer": "Data licensed for academic use only.",
        "history": [{"carbonIntensity": 120}],
    }
    session = MagicMock()
    session.get.return_value = _fake_response(200, payload)
    client = _client(session)
    result = client.get_carbon_intensity_past_range(
        "BE",
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 2, tzinfo=timezone.utc),
    )
    assert result == payload


# --- retry behavior ----------------------------------------------------------


def test_retries_on_429_then_succeeds():
    session = MagicMock()
    session.get.side_effect = [
        _fake_response(429, text="rate limited"),
        _fake_response(200, {"history": []}),
    ]
    client = _client(session)
    result = client.get_carbon_intensity_past_range(
        "BE",
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 2, tzinfo=timezone.utc),
    )
    assert result == {"history": []}
    assert session.get.call_count == 2


def test_retries_on_5xx_then_succeeds():
    session = MagicMock()
    session.get.side_effect = [
        _fake_response(500, text="boom"),
        _fake_response(503, text="unavailable"),
        _fake_response(200, {"history": []}),
    ]
    client = _client(session)
    client.get_carbon_intensity_past_range(
        "BE",
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 2, tzinfo=timezone.utc),
    )
    assert session.get.call_count == 3


def test_retries_on_connection_error_then_succeeds():
    session = MagicMock()
    session.get.side_effect = [
        requests.ConnectionError("network down"),
        _fake_response(200, {"history": []}),
    ]
    client = _client(session)
    client.get_carbon_intensity_past_range(
        "BE",
        datetime(2024, 1, 1, tzinfo=timezone.utc),
        datetime(2024, 1, 2, tzinfo=timezone.utc),
    )
    assert session.get.call_count == 2


def test_retry_exhaustion_raises():
    session = MagicMock()
    session.get.return_value = _fake_response(500, text="boom")
    client = _client(session, max_retries=2)
    with pytest.raises(EMAPIError, match="500 after 2 retries"):
        client.get_carbon_intensity_past_range(
            "BE",
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
    assert session.get.call_count == 3  # initial + 2 retries


# --- non-retryable errors ----------------------------------------------------


def test_401_does_not_retry():
    session = MagicMock()
    session.get.return_value = _fake_response(401, text="unauthorized")
    client = _client(session)
    with pytest.raises(EMAPIError, match="unauthorized"):
        client.get_carbon_intensity_past_range(
            "BE",
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
    assert session.get.call_count == 1


def test_400_does_not_retry():
    session = MagicMock()
    session.get.return_value = _fake_response(400, text="bad zone")
    client = _client(session)
    with pytest.raises(EMAPIError, match="400"):
        client.get_carbon_intensity_past_range(
            "BE",
            datetime(2024, 1, 1, tzinfo=timezone.utc),
            datetime(2024, 1, 2, tzinfo=timezone.utc),
        )
    assert session.get.call_count == 1
