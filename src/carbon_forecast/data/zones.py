"""Zone metadata: Electricity Maps keys and single representative coordinates.

Locked decision: one coordinate per zone, picked as the axis-aligned
bounding-box centroid of the upstream EM zone polygon (from
electricitymap-contrib's `geo/world.geojson`). Coordinates live in
`_zone_centroids.json`, refreshed by `scripts/refresh_zone_centroids.py`.

The single-centroid choice is deliberate (CarbonCast/EnsembleCI lineage).
Accuracy is expected to degrade with zone size and weather heterogeneity;
multi-coordinate aggregation on US-MIDA-PJM is a named stretch experiment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


_CENTROIDS_CACHE = Path(__file__).resolve().parent / "_zone_centroids.json"


@dataclass(frozen=True)
class Zone:
    em_key: str
    latitude: float
    longitude: float
    display_name: str


_ZONE_DISPLAY_NAMES: dict[str, str] = {
    "BE":          "Belgium",
    "FI":          "Finland",
    "SG":          "Singapore",
    "US-MIDA-PJM": "PJM Interconnection",
    "US-NY-NYIS":  "New York ISO",
}


def _load_zones() -> tuple[Zone, ...]:
    if not _CENTROIDS_CACHE.exists():
        raise RuntimeError(
            f"missing centroid cache at {_CENTROIDS_CACHE}. "
            "Run `uv run python scripts/refresh_zone_centroids.py` to regenerate."
        )
    payload = json.loads(_CENTROIDS_CACHE.read_text(encoding="utf-8"))
    centroids = payload["centroids"]
    out = []
    for key, display in _ZONE_DISPLAY_NAMES.items():
        if key not in centroids:
            raise RuntimeError(
                f"zone {key!r} missing from centroid cache; refresh the cache."
            )
        c = centroids[key]
        out.append(Zone(em_key=key, latitude=c["latitude"], longitude=c["longitude"], display_name=display))
    return tuple(out)


ZONES: tuple[Zone, ...] = _load_zones()
_ZONES_BY_KEY: dict[str, Zone] = {z.em_key: z for z in ZONES}


def get_zone(em_key: str) -> Zone:
    try:
        return _ZONES_BY_KEY[em_key]
    except KeyError as exc:
        raise KeyError(
            f"unknown zone {em_key!r}; known zones: {sorted(_ZONES_BY_KEY)}"
        ) from exc
