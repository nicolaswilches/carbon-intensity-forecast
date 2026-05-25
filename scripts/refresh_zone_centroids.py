"""Refresh src/carbon_forecast/data/_zone_centroids.json from upstream.

Fetches electricitymap-contrib's geo/world.geojson, walks the polygons of
each project zone, computes the axis-aligned bounding-box centroid, and
writes a deterministic JSON cache that zones.py loads at import time.

Usage:
    uv run python scripts/refresh_zone_centroids.py

Network is only touched here, not at import time. Run once when the zone
set changes or when the upstream zone boundaries update materially.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

WORLD_GEOJSON_URL = (
    "https://raw.githubusercontent.com/electricitymaps/"
    "electricitymap-contrib/master/geo/world.geojson"
)

PROJECT_ZONES: tuple[str, ...] = (
    "BE",
    "FI",
    "SG",
    "US-MIDA-PJM",
    "US-NY-NYIS",
)

CACHE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/carbon_forecast/data/_zone_centroids.json"
)


def _iter_coords(geometry: dict) -> list[tuple[float, float]]:
    """Walk a GeoJSON Polygon or MultiPolygon and yield every (lon, lat) pair."""
    gtype = geometry["type"]
    coords = geometry["coordinates"]
    out: list[tuple[float, float]] = []

    if gtype == "Polygon":
        rings = coords
        for ring in rings:
            for lon, lat in ring:
                out.append((lon, lat))
    elif gtype == "MultiPolygon":
        for polygon in coords:
            for ring in polygon:
                for lon, lat in ring:
                    out.append((lon, lat))
    else:
        raise ValueError(f"unsupported geometry type: {gtype}")

    return out


def bbox_centroid(geometry: dict) -> tuple[float, float]:
    """Return (latitude, longitude) of the axis-aligned bounding-box center."""
    coords = _iter_coords(geometry)
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return (
        (min(lats) + max(lats)) / 2.0,
        (min(lons) + max(lons)) / 2.0,
    )


def fetch_world_geojson(url: str = WORLD_GEOJSON_URL) -> dict:
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.load(resp)


def compute_centroids(world: dict, zone_keys: tuple[str, ...] = PROJECT_ZONES) -> dict[str, dict]:
    by_zone = {f["properties"]["zoneName"]: f for f in world["features"]}
    out: dict[str, dict] = {}
    for key in zone_keys:
        feature = by_zone.get(key)
        if feature is None:
            raise KeyError(f"zone {key!r} not found in upstream world.geojson")
        lat, lon = bbox_centroid(feature["geometry"])
        out[key] = {"latitude": round(lat, 4), "longitude": round(lon, 4)}
    return out


def main() -> int:
    print(f"Fetching {WORLD_GEOJSON_URL}", file=sys.stderr)
    world = fetch_world_geojson()
    centroids = compute_centroids(world)

    payload = {
        "source": WORLD_GEOJSON_URL,
        "method": "axis-aligned bounding-box centroid of upstream zone polygon",
        "centroids": centroids,
    }
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Deterministic JSON: sorted keys, fixed indent, trailing newline.
    CACHE_PATH.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Wrote {CACHE_PATH}", file=sys.stderr)
    for zone, c in centroids.items():
        print(f"  {zone:<14}  ({c['latitude']:>+7.3f}, {c['longitude']:>+8.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
