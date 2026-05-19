"""Historical extraction orchestration.

Importable functions wired by `scripts/extract_historical.py`. Kept here
so the orchestration logic is unit-testable against mocked clients
without going through argparse.

Behavior:
- Monthly buckets per the storage path convention.
- Idempotent: skip months whose target Parquet file already exists.
- Continue-on-error per month: log and record failures, don't crash the run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator

import pandas as pd

from .em_client import EMAPIError, EMClient, SandboxModeError
from .storage import (
    em_history_path,
    flatten_em_carbon_intensity,
    flatten_em_electricity_flows,
    flatten_em_power_breakdown,
    flatten_weather_hourly,
    weather_archive_path,
    write_parquet_atomic,
)
from .weather_client import WeatherAPIError, WeatherClient
from .zones import Zone

logger = logging.getLogger(__name__)


EM_ENDPOINTS: tuple[str, ...] = (
    "carbon-intensity/past-range",
    "power-breakdown/past-range",
    "electricity-flows/past-range",
)

EM_ENDPOINT_FLATTENERS = {
    "carbon-intensity/past-range": flatten_em_carbon_intensity,
    "power-breakdown/past-range": flatten_em_power_breakdown,
    "electricity-flows/past-range": flatten_em_electricity_flows,
}

EM_ENDPOINT_METHODS = {
    "carbon-intensity/past-range": "get_carbon_intensity_past_range",
    "power-breakdown/past-range": "get_power_breakdown_past_range",
    "electricity-flows/past-range": "get_electricity_flows_past_range",
}

# EM /past-range enforces a 10-day window cap on hourly resolution.
# Months are pulled as multiple sub-windows and concatenated before write.
EM_PAST_RANGE_MAX_DAYS = 10


ProgressCallback = Callable[[str], None]


@dataclass
class ExtractionReport:
    months_pulled: int = 0
    months_skipped: int = 0
    records_written: int = 0
    months_failed: list[tuple[str, str, int, int, str]] = field(default_factory=list)

    def merge(self, other: "ExtractionReport") -> None:
        self.months_pulled += other.months_pulled
        self.months_skipped += other.months_skipped
        self.records_written += other.records_written
        self.months_failed.extend(other.months_failed)


# --- month iteration ---------------------------------------------------------


def iter_months(start: date, end: date) -> Iterator[tuple[int, int]]:
    """Yield (year, month) tuples inclusive from start to end."""
    if end < start:
        raise ValueError(f"end ({end}) must be >= start ({start}).")
    y, m = start.year, start.month
    end_y, end_m = end.year, end.month
    while (y, m) <= (end_y, end_m):
        yield (y, m)
        m += 1
        if m == 13:
            m = 1
            y += 1


def month_window(year: int, month: int) -> tuple[datetime, datetime]:
    """[YYYY-MM-01 00:00 UTC, next-month-01 00:00 UTC)."""
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    if month == 12:
        end = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(year, month + 1, 1, tzinfo=timezone.utc)
    return start, end


def iter_sub_windows(
    start: datetime, end: datetime, max_days: int = EM_PAST_RANGE_MAX_DAYS
) -> Iterator[tuple[datetime, datetime]]:
    """Yield half-open (sub_start, sub_end) pairs covering [start, end), each
    spanning at most `max_days` days. Used to fit EM's /past-range hourly
    window cap. Empty intervals yield nothing.
    """
    if end <= start:
        return
    step = timedelta(days=max_days)
    cursor = start
    while cursor < end:
        nxt = min(cursor + step, end)
        yield cursor, nxt
        cursor = nxt


# Extacts Historical data for carbon-intensity, genenation/power-breakdown, and flows


def extract_em_history(
    em_client: EMClient,
    *,
    zones: Iterable[Zone],
    endpoints: Iterable[str],
    start: date,
    end: date,
    data_root: Path,
    dry_run: bool = False,
    on_progress: ProgressCallback | None = None,
) -> ExtractionReport:
    """Pull (zone, endpoint, month) cells, skip existing, log+continue on errors."""
    endpoints = list(endpoints)
    for endpoint in endpoints:
        if endpoint not in EM_ENDPOINT_FLATTENERS:  # validate the endpoints
            raise ValueError(
                f"unknown EM endpoint {endpoint!r}; known: {sorted(EM_ENDPOINT_FLATTENERS)}"
            )

    report = ExtractionReport()  # create instance of extraction report class
    months = list(iter_months(start, end))

    for zone in zones:  # inititate extraction for each zone
        for endpoint in endpoints:  # loop over each endpoint: ci, power, flows
            flatten = EM_ENDPOINT_FLATTENERS[endpoint]
            method = getattr(em_client, EM_ENDPOINT_METHODS[endpoint])
            for year, month in months:  # extract by month
                target = em_history_path(zone.em_key, endpoint, year, month, data_root)
                tag = f"[EM/{zone.em_key}/{endpoint}/{year:04d}-{month:02d}]"

                if target.exists():
                    report.months_skipped += 1
                    logger.debug("%s skip (exists)", tag)
                    continue

                if dry_run:
                    _emit(on_progress, f"{tag} would pull -> {target}")
                    report.months_pulled += 1
                    continue

                window_start, window_end = month_window(year, month)
                try:
                    frames = []
                    for sub_start, sub_end in iter_sub_windows(window_start, window_end):
                        payload = method(zone.em_key, sub_start, sub_end)
                        frames.append(flatten(payload))
                    df = pd.concat(frames) if frames else flatten({"history": []})
                    df = df.sort_index()
                    df = df[~df.index.duplicated(keep="last")]
                    write_parquet_atomic(df, target)
                except (EMAPIError, SandboxModeError) as exc:
                    msg = f"{type(exc).__name__}: {exc}"
                    logger.warning("%s FAILED %s", tag, msg)
                    report.months_failed.append(
                        (zone.em_key, endpoint, year, month, msg)
                    )
                    continue

                report.months_pulled += 1
                report.records_written += len(df)
                _emit(on_progress, f"{tag} {len(df)} records -> {target}")

    return report


# --- weather extraction ------------------------------------------------------


def extract_weather_history(
    weather_client: WeatherClient,
    *,
    zones: Iterable[Zone],
    start: date,
    end: date,
    data_root: Path,
    dry_run: bool = False,
    on_progress: ProgressCallback | None = None,
) -> ExtractionReport:
    """Pull (zone, month) cells from Open-Meteo archive."""
    report = ExtractionReport()
    months = list(iter_months(start, end))

    for zone in zones:
        for year, month in months:
            target = weather_archive_path(zone.em_key, year, month, data_root)
            tag = f"[Weather/{zone.em_key}/{year:04d}-{month:02d}]"

            if target.exists():
                report.months_skipped += 1
                logger.debug("%s skip (exists)", tag)
                continue

            if dry_run:
                _emit(on_progress, f"{tag} would pull -> {target}")
                report.months_pulled += 1
                continue

            window_start, window_end = month_window(year, month)
            # Open-Meteo archive uses inclusive day windows. End of window
            # is exclusive at the hour boundary; subtract a day to get the
            # final calendar day to include.
            archive_start = window_start.date()
            archive_end = (window_end - timedelta(days=1)).date()
            try:
                payload = weather_client.get_archive(
                    zone.latitude, zone.longitude, archive_start, archive_end
                )
                df = flatten_weather_hourly(payload)
                write_parquet_atomic(df, target)
            except WeatherAPIError as exc:
                msg = f"{type(exc).__name__}: {exc}"
                logger.warning("%s FAILED %s", tag, msg)
                report.months_failed.append((zone.em_key, "weather", year, month, msg))
                continue

            report.months_pulled += 1
            report.records_written += len(df)
            _emit(on_progress, f"{tag} {len(df)} records -> {target}")

    return report


# --- internals ---------------------------------------------------------------


def _emit(cb: ProgressCallback | None, line: str) -> None:
    if cb is not None:
        cb(line)
    logger.info(line)
