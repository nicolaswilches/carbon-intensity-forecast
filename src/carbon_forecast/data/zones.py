"""Zone metadata: Electricity Maps keys and single representative coordinates.

Locked decision: one coordinate per zone, picked as the centroid of the EM
zone bounding box (electricitymap-contrib). Coordinates below are first-cut
estimates; verify against the canonical bounding boxes before Week 2.

The single-centroid choice is deliberate (CarbonCast/EnsembleCI lineage).
Accuracy is expected to degrade with zone size and weather heterogeneity;
multi-coordinate aggregation on US-MIDA-PJM is a named stretch experiment.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Zone:
    em_key: str
    latitude: float
    longitude: float
    display_name: str


# TODO(week-2): verify each (latitude, longitude) against electricitymap-contrib
# bounding-box centroids. https://github.com/electricitymaps/electricitymap-contrib
ZONES: tuple[Zone, ...] = (
    Zone("BE", 50.5, 4.5, "Belgium"),
    Zone("FI", 64.0, 26.0, "Finland"),
    Zone("SG", 1.35, 103.85, "Singapore"),
    Zone("US-MIDA-PJM", 40.0, -77.0, "PJM Interconnection"),
    Zone("US-NY-NYIS", 43.0, -75.0, "New York ISO"),
)

_ZONES_BY_KEY: dict[str, Zone] = {z.em_key: z for z in ZONES}


def get_zone(em_key: str) -> Zone:
    try:
        return _ZONES_BY_KEY[em_key]
    except KeyError as exc:
        raise KeyError(
            f"unknown zone {em_key!r}; known zones: {sorted(_ZONES_BY_KEY)}"
        ) from exc
