"""Open-Meteo weather client.

Hybrid setup per locked decisions:
- ERA5 reanalysis via the archive endpoint for training history.
- GFS via /v1/gfs for forecast inference during the Test B window.

Single coordinate per zone (centroid). No API key. Returns raw JSON dicts;
processor.py derives u/v from speed/direction and joins to the zone series.
"""

from __future__ import annotations

import logging
import random
import time
from datetime import date
from typing import Any

import requests

logger = logging.getLogger(__name__)

DEFAULT_ARCHIVE_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
DEFAULT_FORECAST_BASE_URL = "https://api.open-meteo.com/v1/gfs"

# CarbonCast-aligned variable set. `shortwave_radiation` is the surface
# global horizontal irradiance, the Open-Meteo name for DSWRF.
HOURLY_VARIABLES: list[str] = [
    "temperature_2m",
    "dewpoint_2m",
    "shortwave_radiation",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
]

MAX_FORECAST_DAYS = 16
MIN_LAT, MAX_LAT = -90.0, 90.0
MIN_LON, MAX_LON = -180.0, 180.0


class WeatherAPIError(RuntimeError):
    """Non-retryable Open-Meteo API failure."""


class WeatherClient:
    """Open-Meteo v1 client (archive + GFS)."""

    def __init__(
        self,
        archive_base_url: str = DEFAULT_ARCHIVE_BASE_URL,
        forecast_base_url: str = DEFAULT_FORECAST_BASE_URL,
        timeout: float = 30.0,
        max_retries: int = 5,
        backoff_base: float = 1.0,
        session: requests.Session | None = None,
    ) -> None:
        self.archive_base_url = archive_base_url
        self.forecast_base_url = forecast_base_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base

        self._session = session if session is not None else requests.Session()

    # --- public endpoints ----------------------------------------------------

    def get_archive(
        self,
        latitude: float,
        longitude: float,
        start: date,
        end: date,
        variables: list[str] | None = None,
    ) -> dict[str, Any]:
        _validate_coords(latitude, longitude)
        if start > end:
            raise ValueError(f"start ({start}) must be <= end ({end}).")
        params = {
            "latitude": str(latitude),
            "longitude": str(longitude),
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "hourly": ",".join(variables or HOURLY_VARIABLES),
            "timezone": "UTC",
        }
        return self._get(self.archive_base_url, params)

    def get_gfs_forecast(
        self,
        latitude: float,
        longitude: float,
        forecast_days: int = 4,
        past_days: int = 0,
        variables: list[str] | None = None,
    ) -> dict[str, Any]:
        _validate_coords(latitude, longitude)
        if not 1 <= forecast_days <= MAX_FORECAST_DAYS:
            raise ValueError(
                f"forecast_days must be in [1, {MAX_FORECAST_DAYS}], got {forecast_days}."
            )
        if past_days < 0:
            raise ValueError(f"past_days must be >= 0, got {past_days}.")
        params = {
            "latitude": str(latitude),
            "longitude": str(longitude),
            "forecast_days": str(forecast_days),
            "past_days": str(past_days),
            "hourly": ",".join(variables or HOURLY_VARIABLES),
            "timezone": "UTC",
        }
        return self._get(self.forecast_base_url, params)

    # --- internals -----------------------------------------------------------

    def _get(self, url: str, params: dict[str, str]) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._session.get(url, params=params, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                if attempt == self.max_retries:
                    raise WeatherAPIError(
                        f"network failure after {self.max_retries} retries: {exc}"
                    ) from exc
                self._sleep_backoff(attempt)
                continue

            if response.status_code == 200:
                return response.json()

            if response.status_code == 429 or 500 <= response.status_code < 600:
                if attempt == self.max_retries:
                    raise WeatherAPIError(
                        f"{response.status_code} after {self.max_retries} retries on {url}: "
                        f"{response.text[:200]}"
                    )
                logger.warning(
                    "Open-Meteo %s on %s, retry %d/%d",
                    response.status_code,
                    url,
                    attempt + 1,
                    self.max_retries,
                )
                self._sleep_backoff(attempt)
                continue

            raise WeatherAPIError(
                f"{response.status_code} on {url}: {response.text[:200]}"
            )

        raise WeatherAPIError(f"unreachable retry exhaustion: {last_exc}")

    def _sleep_backoff(self, attempt: int) -> None:
        delay = self.backoff_base * (2**attempt) + random.uniform(0, 0.5)
        time.sleep(delay)


# --- module-level helpers ----------------------------------------------------


def _validate_coords(latitude: float, longitude: float) -> None:
    if not MIN_LAT <= latitude <= MAX_LAT:
        raise ValueError(f"latitude {latitude} out of range [{MIN_LAT}, {MAX_LAT}].")
    if not MIN_LON <= longitude <= MAX_LON:
        raise ValueError(f"longitude {longitude} out of range [{MIN_LON}, {MAX_LON}].")
