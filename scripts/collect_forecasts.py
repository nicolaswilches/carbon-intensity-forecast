"""Live forecast-snapshot collector for the Test B head-to-head window.

Captures, per zone, at each invocation:
  - EM's operational CI forecast (the Contribution 4 benchmark target).
  - The GFS weather forecast over the full model horizon, so the open model
    can be replayed offline on the exact inputs it would have seen now.

Both are append-only snapshots keyed by capture time:
  data/raw/em/forecasts/{zone}/{snapshot_iso}.parquet
  data/raw/weather/forecasts/{zone}/{snapshot_iso}.parquet

Design notes:
  - Idempotent per tick: a given snapshot timestamp writes one file per zone.
  - Continues past per-zone failures; reports them and exits non-zero so a
    scheduler (launchd/cron) surfaces partial failures in logs.
  - Self-contained env loading (dotenv) so launchd can run it with no shell
    profile. Ground-truth backfill is a separate weekly past-range job.

Usage:
    python scripts/collect_forecasts.py
    python scripts/collect_forecasts.py --zones BE FI --forecast-days 4
    python scripts/collect_forecasts.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from carbon_forecast.data import storage
from carbon_forecast.data.em_client import EMClient
from carbon_forecast.data.weather_client import WeatherClient
from carbon_forecast.data.zones import ZONES, get_zone

logger = logging.getLogger("collect_forecasts")

# Open model forecasts the full 96h; pull 4 GFS days to cover it with margin.
DEFAULT_FORECAST_DAYS = 4
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--zones",
        nargs="+",
        default=[z.em_key for z in ZONES],
        help="EM zone keys (default: all locked zones)",
    )
    p.add_argument(
        "--forecast-days",
        type=int,
        default=DEFAULT_FORECAST_DAYS,
        help=f"GFS forecast days to snapshot (default: {DEFAULT_FORECAST_DAYS})",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help="data/ root (default: project data/)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="fetch and log shapes but write nothing to disk",
    )
    return p


def _collect_zone(
    em: EMClient,
    weather: WeatherClient,
    zone_key: str,
    snapshot: datetime,
    forecast_days: int,
    data_root: Path,
    dry_run: bool,
) -> None:
    """Snapshot EM CI forecast + GFS weather for one zone. Raises on failure."""
    zone = get_zone(zone_key)

    # EM operational CI forecast.
    ci_payload = em.get_carbon_intensity_forecast(zone_key)
    ci_df = storage.flatten_em_carbon_intensity_forecast(ci_payload)
    ci_df["snapshot_utc"] = snapshot

    # GFS weather forecast over the model horizon.
    wx_payload = weather.get_gfs_forecast(
        latitude=zone.latitude,
        longitude=zone.longitude,
        forecast_days=forecast_days,
    )
    wx_df = storage.flatten_weather_hourly(wx_payload)
    wx_df["snapshot_utc"] = snapshot

    logger.info(
        "%-14s CI forecast: %d rows (to %s) | weather: %d rows",
        zone_key,
        len(ci_df),
        ci_df.index.max() if len(ci_df) else "n/a",
        len(wx_df),
    )

    if dry_run:
        return

    ci_path = storage.em_forecast_path(zone_key, snapshot, data_root)
    wx_path = storage.weather_forecast_path(zone_key, snapshot, data_root)
    storage.write_parquet_atomic(ci_df, ci_path)
    storage.write_parquet_atomic(wx_df, wx_path)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = _build_parser().parse_args(argv)
    load_dotenv(PROJECT_ROOT / ".env")

    em = EMClient()
    weather = WeatherClient()
    # One capture time for the whole tick, so all zones share a snapshot key.
    snapshot = datetime.now(timezone.utc).replace(microsecond=0)

    logger.info(
        "collecting forecasts at %s for %d zones (dry_run=%s)",
        snapshot.isoformat(),
        len(args.zones),
        args.dry_run,
    )

    failures: list[tuple[str, str]] = []
    for zone_key in args.zones:
        try:
            _collect_zone(
                em, weather, zone_key, snapshot, args.forecast_days,
                args.data_root, args.dry_run,
            )
        except Exception as exc:  # continue-on-error: one zone must not sink the tick
            logger.exception("zone %s failed: %s", zone_key, exc)
            failures.append((zone_key, str(exc)))

    ok = len(args.zones) - len(failures)
    logger.info("done: %d/%d zones captured", ok, len(args.zones))
    if failures:
        for zone_key, msg in failures:
            logger.error("FAILED %s: %s", zone_key, msg)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
