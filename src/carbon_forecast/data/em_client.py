"""Electricity Maps API client.

- Thin HTTP wrapper around the v3 API. Returns parsed JSON dicts
- Downstream modules flatten to DataFrames and persist. 
- Detects sandbox-mode responses and refuses to return them (see LESSONS.md).
"""

# Imports & contants
from __future__ import annotations

import logging
import os
import random
import time
from datetime import datetime, timezone
from typing import Any

import requests

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.electricitymap.org/v3"
SANDBOX_DISCLAIMER_MARKER = "SANDBOX MODE"
SANDBOX_ESTIMATION_METHOD = "SANDBOX_MODE_DATA"

ENDPOINT_CARBON_INTENSITY_PAST_RANGE = "carbon-intensity/past-range"
ENDPOINT_POWER_BREAKDOWN_PAST_RANGE = "power-breakdown/past-range"
ENDPOINT_ELECTRICITY_FLOWS_PAST_RANGE = "electricity-flows/past-range"

# Operational forecast endpoint (Test B head-to-head, Contribution 4).
# Academic key returns a ~24h CI horizon, hourly, consumption-based; the
# power-breakdown forecast endpoint is not licensed on this key.
ENDPOINT_CARBON_INTENSITY_FORECAST = "carbon-intensity/forecast"


# Exception class for scenarios with non-retryable API failures
class EMAPIError(RuntimeError):
    """Non-retryable Electricity Maps API failure (4xx other than 429, bad payload, etc.)."""

# Exception class for scenarios where sandbox mode is detected
class SandboxModeError(RuntimeError):
    """Response contains sandbox-mode markers; refuse to persist synthetic data."""


class EMClient:
    """Electricity Maps v3 API client.

    Designed as a single-responsibility HTTP layer:
    - Auth via the `auth-token` header
    - Retry with exponential backoff on 429 and 5xx
    - Sandbox-mode detection on every 200 response
    - Returns the raw parsed JSON dict, no DataFrame conversion
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        session: requests.Session | None = None,
    ) -> None:
        resolved_key = api_key if api_key is not None else os.environ.get("EM_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "EM_API_KEY missing. Pass api_key=... or set the env var (source .env)."
            )

        self._api_key = resolved_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._session = session if session is not None else requests.Session()

    # public methods / endpoints

    def get_carbon_intensity_past_range( # carbon intensity
        self, zone: str, start: datetime, end: datetime
    ) -> dict[str, Any]:
        return self._past_range(ENDPOINT_CARBON_INTENSITY_PAST_RANGE, zone, start, end)

    def get_power_breakdown_past_range( # power generation
        self, zone: str, start: datetime, end: datetime
    ) -> dict[str, Any]:
        return self._past_range(ENDPOINT_POWER_BREAKDOWN_PAST_RANGE, zone, start, end)

    def get_electricity_flows_past_range( # electricity flows
        self, zone: str, start: datetime, end: datetime
    ) -> dict[str, Any]:
        return self._past_range(ENDPOINT_ELECTRICITY_FLOWS_PAST_RANGE, zone, start, end)

    def get_carbon_intensity_forecast( # operational CI forecast (live snapshot)
        self, zone: str
    ) -> dict[str, Any]:
        """Latest operational CI forecast for a zone.

        No time window: EM returns the forecast from the current hour out to
        the licensed horizon (24h on the academic key). Same retry, sandbox,
        and auth handling as the past-range calls.
        """
        return self._get(ENDPOINT_CARBON_INTENSITY_FORECAST, {"zone": zone})

    # internal methods

    def _past_range( # validates the time window and passes result to ._get()
        self, endpoint: str, zone: str, start: datetime, end: datetime
    ) -> dict[str, Any]:
        _validate_window(start, end)
        params = {
            "zone": zone,
            "start": _to_iso_z(start),
            "end": _to_iso_z(end),
        }
        return self._get(endpoint, params)

    def _get(self, endpoint: str, params: dict[str, str]) -> dict[str, Any]:
        url = f"{self.base_url}/{endpoint}"
        headers = {"auth-token": self._api_key}

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1): # tries to call for N times
            try:
                response = self._session.get( # call
                    url, params=params, headers=headers, timeout=self.timeout
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    raise EMAPIError(
                        f"network failure after {self.max_retries} retries: {exc}"
                    ) from exc
                self._sleep_backoff(attempt)
                continue

            if response.status_code == 200: # call success
                payload = response.json()
                _check_sandbox(payload)
                return payload

            if response.status_code == 429 or 500 <= response.status_code < 600:
                if attempt == self.max_retries:
                    raise EMAPIError(
                        f"{response.status_code} after {self.max_retries} retries on {endpoint}: "
                        f"{response.text[:200]}"
                    )
                logger.warning(
                    "EM %s on %s, retry %d/%d",
                    response.status_code,
                    endpoint,
                    attempt + 1,
                    self.max_retries,
                )
                self._sleep_backoff(attempt)
                continue

            if response.status_code in (401, 403):
                raise EMAPIError(
                    f"{response.status_code} unauthorized on {endpoint}. "
                    "Check EM_API_KEY and that your tier covers this endpoint/zone."
                )

            raise EMAPIError(
                f"{response.status_code} on {endpoint}: {response.text[:200]}"
            )

        # Defensive — loop should always return or raise above.
        raise EMAPIError(f"unreachable retry exhaustion: {last_exc}")

    def _sleep_backoff(self, attempt: int) -> None:
        delay = self.backoff_base * (2**attempt) + random.uniform(0, 0.5)
        time.sleep(delay)


# module-level helper functions 


def _to_iso_z(dt: datetime) -> str:
    """Format an aware UTC datetime as ISO-8601 with Z suffix, no microseconds."""
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _validate_window(start: datetime, end: datetime) -> None:
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end must be timezone-aware datetimes (UTC).")
    if start >= end:
        raise ValueError(f"start ({start}) must be strictly before end ({end}).")


def _check_sandbox(payload: dict[str, Any]) -> None:
    disclaimer = payload.get("_disclaimer", "")
    if isinstance(disclaimer, str) and SANDBOX_DISCLAIMER_MARKER in disclaimer:
        raise SandboxModeError(
            "EM response carries a sandbox disclaimer; refusing to return synthetic data. "
            "Verify the academic-tier key is active."
        )

    for key in ("history", "data", "forecast"):
        records = payload.get(key)
        if isinstance(records, list):
            for record in records:
                if (
                    isinstance(record, dict)
                    and record.get("estimationMethod") == SANDBOX_ESTIMATION_METHOD
                ):
                    raise SandboxModeError(
                        f"EM response contains SANDBOX_MODE_DATA records in '{key}'."
                    )
