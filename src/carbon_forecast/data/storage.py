"""
Storage layer: raw JSONs to DataFrame, atomic Parquet I/O.

Path conventions (amended 2026-05-19, monthly buckets):
- EM historical:        data/raw/em/{zone}/{endpoint}/{year}-{month:02d}.parquet
- EM forecast snapshot: data/raw/em/forecasts/{zone}/{snapshot_iso}.parquet
- Weather archive:      data/raw/weather/{zone}/{year}-{month:02d}.parquet
- Processed:            data/processed/{zone}.parquet

Flatten functions normalize to a wide DataFrame with a UTC DatetimeIndex
named "datetime". Column names follow project-wide conventions (snake_case
zone codes, unit suffixes, prod_/cons_/import_/export_ prefixes for
breakdown payloads). No feature engineering happens here.
"""

from __future__ import annotations

import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


# --- path helpers ------------------------------------------------------------


def em_history_path(
    zone: str, endpoint: str, year: int, month: int, root: Path
) -> Path:
    """Path for a monthly EM historical Parquet file."""
    return (
        Path(root) / "raw" / "em" / zone / endpoint / f"{year:04d}-{month:02d}.parquet"
    )


def em_forecast_path(zone: str, snapshot: datetime, root: Path) -> Path:
    """Path for a single EM forecast snapshot. snapshot must be tz-aware UTC."""
    if snapshot.tzinfo is None:
        raise ValueError("snapshot must be timezone-aware (UTC).")
    iso = snapshot.astimezone(tz=snapshot.tzinfo).strftime("%Y%m%dT%H%M%SZ")
    return Path(root) / "raw" / "em" / "forecasts" / zone / f"{iso}.parquet"


def weather_archive_path(zone: str, year: int, month: int, root: Path) -> Path:
    """Path for a monthly weather archive Parquet file."""
    return Path(root) / "raw" / "weather" / zone / f"{year:04d}-{month:02d}.parquet"


def processed_path(zone: str, root: Path) -> Path:
    """Path for the processed, modeling-ready Parquet of a zone."""
    return Path(root) / "processed" / f"{zone}.parquet"


# --- flatten: EM payloads ----------------------------------------------------


def flatten_em_carbon_intensity(payload: dict[str, Any]) -> pd.DataFrame:
    """
    Carbon-intensity history -> single-column DataFrame.

    EM payload shape:
        {"zone": "BE",
        "history": [{"datetime": "...Z", "carbonIntensity": 120, "estimationMethod": ..., ...},
        ...
      ]}
    Output columns: carbon_intensity_gco2eq_kwh (Float64).
    """
    history = payload.get("history") or []  #
    if not history:
        return _empty_frame(
            ["carbon_intensity_gco2eq_kwh"]
        )  # returns empty df with shape

    rows = []
    for record in history:
        rows.append(  # appends a dictionary with the datime and the ci record
            {
                "datetime": record["datetime"],
                "carbon_intensity_gco2eq_kwh": record.get("carbonIntensity"),
            }
        )
    return _finalize(pd.DataFrame(rows))  # converts the list 'rows' into a dataframe


def flatten_em_power_breakdown(payload: dict[str, Any]) -> pd.DataFrame:
    """
    Power-breakdown history -> wide DataFrame (production and consumption mix).

    EM payload exposes two source-keyed nested dicts per timestamp:
      - powerProductionBreakdown   -> prod_<source>_mw
      - powerConsumptionBreakdown  -> cons_<source>_mw
    Plus two top-level scalars carried through for cross-check:
      - powerImportTotal           -> import_total_mw
      - powerExportTotal           -> export_total_mw

    Partner-aggregated powerImportBreakdown / powerExportBreakdown are
    intentionally NOT flattened here. Per-partner flow data is the job
    of `flatten_em_power_flows`, which is the authoritative source for
    the flow signal (Contribution 2 of the thesis). Keeping import/export
    breakdown columns out of this table prevents downstream ambiguity
    about which endpoint a column came from.
    """
    history = payload.get("history") or []
    if not history:
        return _empty_frame([])

    rows = []
    for record in history:
        row: dict[str, Any] = {"datetime": record["datetime"]}
        _add_nested(row, record.get("powerProductionBreakdown"), prefix="prod")
        _add_nested(row, record.get("powerConsumptionBreakdown"), prefix="cons")
        if "powerImportTotal" in record:
            row["import_total_mw"] = record["powerImportTotal"]
        if "powerExportTotal" in record:
            row["export_total_mw"] = record["powerExportTotal"]
        rows.append(row)
    return _finalize(pd.DataFrame(rows))


def flatten_em_power_flows(payload: dict[str, Any]) -> pd.DataFrame:
    """
    Power-flows history -> wide DataFrame with import_/export_ columns.

    EM payload shape (per locked decisions; exact field names verified
    on first real pull, fallback to a generic walk if EM renames):
      history items expose flat numeric fields keyed by partner zone, or
      nested powerImports/powerExports dicts. Both shapes handled.
    """
    history = payload.get("history") or []
    if not history:
        return _empty_frame([])

    rows = []
    for record in history:
        row: dict[str, Any] = {"datetime": record["datetime"]}
        if isinstance(record.get("powerImports"), dict):
            _add_nested(row, record["powerImports"], prefix="import")
        if isinstance(record.get("powerExports"), dict):
            _add_nested(row, record["powerExports"], prefix="export")
        rows.append(row)
    return _finalize(pd.DataFrame(rows))


# --- flatten: Open-Meteo ------------------------------------------------------


def flatten_weather_hourly(payload: dict[str, Any]) -> pd.DataFrame:
    """Open-Meteo `hourly` block -> DataFrame indexed by UTC time."""
    hourly = payload.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return _empty_frame([])

    df = pd.DataFrame({k: v for k, v in hourly.items() if k != "time"})
    df["datetime"] = times
    return _finalize(df)


# --- write / read ------------------------------------------------------------


def write_parquet_atomic(df: pd.DataFrame, path: Path) -> Path:
    """Write to a temp file in the same dir, then atomically rename.

    Same-dir temp guarantees the rename is atomic (POSIX rename(2) on the
    same filesystem). A crash mid-write leaves the temp file behind but
    the destination file is never partial. Idempotent: overwrites if the
    destination exists.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        df.to_parquet(tmp_path, index=True)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return path


def read_parquet(path: Path) -> pd.DataFrame:
    """Read a Parquet file written by this module; UTC index preserved."""
    df = pd.read_parquet(path)
    if "datetime" in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.set_index("datetime").sort_index()
    elif isinstance(df.index, pd.DatetimeIndex):
        df.index = (
            df.index.tz_convert("UTC")
            if df.index.tz is not None
            else df.index.tz_localize("UTC")
        )
    return df


def list_months_present(zone: str, endpoint: str, root: Path) -> set[tuple[int, int]]:
    """Return the set of (year, month) tuples already on disk for a (zone, endpoint).

    Used by the extraction script to skip months that are complete.
    """
    base = Path(root) / "raw" / "em" / zone / endpoint
    if not base.exists():
        return set()
    out: set[tuple[int, int]] = set()
    pattern = re.compile(r"^(\d{4})-(\d{2})\.parquet$")
    for entry in base.iterdir():
        m = pattern.match(entry.name)
        if m:
            out.add((int(m.group(1)), int(m.group(2))))
    return out


# --- private helper functions


def _add_nested(
    row: dict[str, Any], nested: dict[str, Any] | None, *, prefix: str
) -> None:
    """Flatten a {key: value} dict onto a row with snake-cased keys."""
    if not isinstance(nested, dict):
        return
    for key, value in nested.items():
        if value is None:
            continue
        row[f"{prefix}_{_snake(str(key))}_mw"] = value


def _snake(s: str) -> str:
    """Lowercase, replace non-alphanumerics with underscores, collapse runs."""
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _empty_frame(columns: Iterable[str]) -> pd.DataFrame:
    """Return an empty DataFrame with a UTC DatetimeIndex named 'datetime'."""
    idx = pd.DatetimeIndex([], tz="UTC", name="datetime")
    return pd.DataFrame({c: pd.Series(dtype="float64") for c in columns}, index=idx)


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the 'datetime' column to UTC, set as index, sort, dedupe."""
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df
