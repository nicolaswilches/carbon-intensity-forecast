"""Tests for zones.py and the centroid cache.

These tests confirm that the cache file produces well-shaped Zone objects
within plausible geographic bounding regions. The exact lat/lon values are
authoritative in `_zone_centroids.json`; tests check membership and ranges.
"""

from __future__ import annotations

from carbon_forecast.data.zones import ZONES, Zone, get_zone


EXPECTED_KEYS = {"BE", "FI", "SG", "US-MIDA-PJM", "US-NY-NYIS"}

# Rough country/region bounding boxes used as sanity checks only.
# (min_lat, max_lat, min_lon, max_lon).
EXPECTED_REGIONS: dict[str, tuple[float, float, float, float]] = {
    "BE":          (49.0, 52.0,    2.0,   7.0),
    "FI":          (59.0, 71.0,   19.0,  32.0),
    "SG":          ( 1.0,  2.0,  103.0, 105.0),
    "US-MIDA-PJM": (35.0, 43.0,  -90.0, -73.0),
    "US-NY-NYIS":  (40.0, 46.0,  -80.0, -71.0),
}


def test_five_zones_loaded():
    assert {z.em_key for z in ZONES} == EXPECTED_KEYS
    assert len(ZONES) == 5


def test_each_zone_is_a_frozen_dataclass():
    for z in ZONES:
        assert isinstance(z, Zone)


def test_centroids_inside_expected_regions():
    for z in ZONES:
        lat_min, lat_max, lon_min, lon_max = EXPECTED_REGIONS[z.em_key]
        assert lat_min <= z.latitude <= lat_max, (
            f"{z.em_key} latitude {z.latitude} outside [{lat_min}, {lat_max}]"
        )
        assert lon_min <= z.longitude <= lon_max, (
            f"{z.em_key} longitude {z.longitude} outside [{lon_min}, {lon_max}]"
        )


def test_get_zone_returns_matching_object():
    z = get_zone("BE")
    assert z.em_key == "BE"
    assert z.display_name == "Belgium"


def test_get_zone_unknown_raises():
    import pytest

    with pytest.raises(KeyError, match="known zones"):
        get_zone("XX-FAKE")


def test_display_names_present_for_all_zones():
    for z in ZONES:
        assert z.display_name and isinstance(z.display_name, str)
