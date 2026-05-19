"""CLI for the historical EM + Open-Meteo extraction.

Usage:
    python scripts/extract_historical.py --start 2021-01 --end 2025-12
    python scripts/extract_historical.py --start 2024-01 --end 2024-03 \\
        --zones BE FI --source em --dry-run

Idempotent: re-running skips months already on disk. Continues past
per-month failures and reports them at the end.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

from carbon_forecast.data.em_client import EMClient
from carbon_forecast.data.extract import (
    EM_ENDPOINTS,
    ExtractionReport,
    extract_em_history,
    extract_weather_history,
)
from carbon_forecast.data.weather_client import WeatherClient
from carbon_forecast.data.zones import ZONES, get_zone

logger = logging.getLogger("extract_historical")


def _parse_month(s: str) -> date:
    """Parse YYYY-MM as the first day of that month."""
    try:
        year, month = s.split("-")
        return date(int(year), int(month), 1)
    except (ValueError, AttributeError) as exc:
        raise argparse.ArgumentTypeError(
            f"expected YYYY-MM, got {s!r}: {exc}"
        ) from exc


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", type=_parse_month, required=True, help="YYYY-MM inclusive")
    p.add_argument("--end", type=_parse_month, required=True, help="YYYY-MM inclusive")
    p.add_argument(
        "--zones",
        nargs="+",
        default=[z.em_key for z in ZONES],
        help="EM zone keys (default: all locked zones)",
    )
    p.add_argument(
        "--endpoints",
        nargs="+",
        default=list(EM_ENDPOINTS),
        help="EM endpoints to pull (default: all three)",
    )
    p.add_argument(
        "--source",
        choices=("em", "weather", "both"),
        default="both",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="root for raw/ and processed/ (default: ./data)",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _print_report(label: str, report: ExtractionReport) -> None:
    print(f"\n== {label} ==")
    print(f"  pulled:  {report.months_pulled}")
    print(f"  skipped: {report.months_skipped}")
    print(f"  rows:    {report.records_written}")
    print(f"  failed:  {len(report.months_failed)}")
    for zone, endpoint, year, month, err in report.months_failed:
        print(f"    - {zone}/{endpoint}/{year:04d}-{month:02d}: {err}")


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    zones = [get_zone(z) for z in args.zones]

    if args.source in ("em", "both"):
        em_client = EMClient()
        em_report = extract_em_history(
            em_client,
            zones=zones,
            endpoints=args.endpoints,
            start=args.start,
            end=args.end,
            data_root=args.data_root,
            dry_run=args.dry_run,
        )
        _print_report("EM", em_report)

    if args.source in ("weather", "both"):
        weather_client = WeatherClient()
        weather_report = extract_weather_history(
            weather_client,
            zones=zones,
            start=args.start,
            end=args.end,
            data_root=args.data_root,
            dry_run=args.dry_run,
        )
        _print_report("Weather", weather_report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
