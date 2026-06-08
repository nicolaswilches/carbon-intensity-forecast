"""Calendar feature derivation.

Cyclical and categorical time features for the Tier 1/Tier 2 models. Computed
in each zone's LOCAL time, because hour-of-day, weekend, and holiday only carry
demand-cycle signal locally. Storage stays UTC (locked convention); this module
localizes only to derive features, then returns a frame aligned to the original
UTC index.

Feature set (locked 2026-05-11): hour-of-day sin/cos, hour-of-year sin/cos,
day-of-week, weekend flag, holiday flag.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import holidays
import numpy as np
import pandas as pd

# Zone -> IANA timezone and the `holidays` country/subdiv used for the holiday
# flag. US-MIDA-PJM spans several states; New York is the consistent anchor for
# both US zones (and matches US-NY-NYIS exactly).
ZONE_LOCALE: dict[str, dict[str, str]] = {
    "BE":          {"tz": "Europe/Brussels",  "country": "BE", "subdiv": ""},
    "FI":          {"tz": "Europe/Helsinki",  "country": "FI", "subdiv": ""},
    "SG":          {"tz": "Asia/Singapore",   "country": "SG", "subdiv": ""},
    "US-MIDA-PJM": {"tz": "America/New_York", "country": "US", "subdiv": "NY"},
    "US-NY-NYIS":  {"tz": "America/New_York", "country": "US", "subdiv": "NY"},
}

CALENDAR_FEATURES: list[str] = [
    "hour_sin", "hour_cos",
    "yearhour_sin", "yearhour_cos",
    "day_of_week", "is_weekend", "is_holiday",
]


def _locale(zone: str) -> dict[str, str]:
    try:
        return ZONE_LOCALE[zone]
    except KeyError as exc:
        raise KeyError(
            f"no locale for zone {zone!r}; known: {sorted(ZONE_LOCALE)}"
        ) from exc


def calendar_features(index: pd.DatetimeIndex, zone: str) -> pd.DataFrame:
    """Build the calendar feature frame for a UTC DatetimeIndex.

    Returns a DataFrame indexed identically to `index` (UTC), with the columns
    in `CALENDAR_FEATURES`. Cyclical features use sin/cos so the model sees
    23:00 and 00:00 as adjacent.
    """
    if index.tz is None:
        raise ValueError("index must be tz-aware UTC.")
    loc = _locale(zone)
    local = index.tz_convert(ZoneInfo(loc["tz"]))

    hour = local.hour.to_numpy()
    # Hour-of-year in [0, 8783]; leap years stretch slightly, immaterial for the cycle.
    yearhour = ((local.dayofyear - 1) * 24 + local.hour).to_numpy()
    dow = local.dayofweek.to_numpy()  # Mon=0 .. Sun=6

    subdiv = loc["subdiv"] or None
    years = range(local.year.min(), local.year.max() + 1)
    cal = holidays.country_holidays(loc["country"], subdiv=subdiv, years=years)
    local_dates = local.date
    is_holiday = np.fromiter((d in cal for d in local_dates), dtype=float, count=len(local_dates))

    out = pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * hour / 24.0),
            "hour_cos": np.cos(2 * np.pi * hour / 24.0),
            "yearhour_sin": np.sin(2 * np.pi * yearhour / 8760.0),
            "yearhour_cos": np.cos(2 * np.pi * yearhour / 8760.0),
            "day_of_week": dow.astype(float),
            "is_weekend": (dow >= 5).astype(float),
            "is_holiday": is_holiday,
        },
        index=index,
    )
    return out
