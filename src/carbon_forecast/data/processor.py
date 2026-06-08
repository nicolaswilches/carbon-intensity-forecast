"""Processor: raw per-endpoint Parquet -> modeling-ready hourly frame per zone.

Joins the four raw streams (carbon intensity, power breakdown, electricity
flows, weather) on a shared hourly UTC index and derives the modeling targets
and features:

  - prod_based_ci_lifecycle / prod_based_ci_direct: CarbonCast Eq. 1 over the
    domestic production mix (imports/exports ignored, per CarbonCast 2.3). The
    lifecycle target is the E2 (CarbonCast-faithful, production-based) label.
  - cons_based_ci: EM's reported consumption-based CI, carried through as the
    E3 target.
  - per-source production (prod_*_mw), total generation, renewable share.
  - per-partner net flow and net total flow.
  - weather: u/v wind derived from speed/direction, plus the raw GFS variables.
  - calendar features (zone-local) from utils.calendar.

No normalization or windowing here; those are separate, train-only steps.

Emission factors: CarbonCast Table 1 (Maji et al., BuildSys '22), g/kWh.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from carbon_forecast.data import storage
from carbon_forecast.utils.calendar import calendar_features

logger = logging.getLogger(__name__)

# CarbonCast Table 1, lifecycle (scope 3) emission factors in gCO2eq/kWh.
EF_LIFECYCLE: dict[str, float] = {
    "coal": 820, "oil": 650, "gas": 490, "nuclear": 12, "solar": 45,
    "wind": 11, "hydro": 24, "other": 700, "biomass": 230, "geothermal": 38,
}
# CarbonCast Table 1, direct (scope 2) emission factors in gCO2eq/kWh.
EF_DIRECT: dict[str, float] = {
    "coal": 760, "oil": 406, "gas": 370, "nuclear": 0, "solar": 0,
    "wind": 0, "hydro": 0, "other": 575, "biomass": 0, "geothermal": 0,
}

# EM power-breakdown source key -> CarbonCast category. Storage discharge is
# mapped to its underlying technology (hydro) / to "other" for batteries, since
# CarbonCast (2022) predates grid-scale battery accounting.
EM_SOURCE_TO_CC: dict[str, str] = {
    "nuclear": "nuclear", "geothermal": "geothermal", "biomass": "biomass",
    "coal": "coal", "wind": "wind", "solar": "solar", "hydro": "hydro",
    "gas": "gas", "oil": "oil", "unknown": "other",
    "hydro_discharge": "hydro", "battery_discharge": "other",
}

# Sources counted as renewable for the renewable-share feature.
RENEWABLE_CC = {"solar", "wind", "hydro", "biomass", "geothermal"}

GFS_WIND_SPEED = "wind_speed_10m"
GFS_WIND_DIR = "wind_direction_10m"


def _load_endpoint(zone: str, endpoint: str, data_root: Path) -> pd.DataFrame:
    """Concatenate every monthly Parquet for one (zone, endpoint)."""
    base = Path(data_root) / "raw" / "em" / zone / endpoint
    files = sorted(base.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no Parquet files under {base}")
    frames = [storage.read_parquet(f) for f in files]
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="last")]


def _load_weather(zone: str, data_root: Path) -> pd.DataFrame:
    base = Path(data_root) / "raw" / "weather" / zone
    files = sorted(base.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no weather Parquet under {base}")
    frames = [storage.read_parquet(f) for f in files]
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="last")]


def production_based_ci(prod: pd.DataFrame, factors: dict[str, float]) -> pd.Series:
    """CarbonCast Eq. 1: sum(E_i * CEF_i) / sum(E_i) over domestic production.

    `prod` holds prod_<source>_mw columns. Sources are mapped to CarbonCast
    categories; negative production (storage charging shown as negative) is
    clipped to 0 so it neither emits nor inflates the denominator.
    """
    numer = pd.Series(0.0, index=prod.index)
    denom = pd.Series(0.0, index=prod.index)
    for col in prod.columns:
        if not col.startswith("prod_") or not col.endswith("_mw"):
            continue
        source = col[len("prod_"):-len("_mw")]
        cc = EM_SOURCE_TO_CC.get(source)
        if cc is None:
            logger.warning("unmapped production source %r; treating as 'other'", source)
            cc = "other"
        energy = prod[col].fillna(0.0).clip(lower=0.0)
        numer = numer + energy * factors[cc]
        denom = denom + energy
    return numer / denom.replace(0.0, np.nan)


def _derive_wind_uv(weather: pd.DataFrame) -> pd.DataFrame:
    """u/v wind components at 10m from speed (m/s) and meteorological direction.

    Direction is the bearing the wind blows FROM; u/v are the vector the wind
    blows TO, matching CarbonCast's u/v-component features.
    """
    if GFS_WIND_SPEED not in weather or GFS_WIND_DIR not in weather:
        return weather
    rad = np.deg2rad(weather[GFS_WIND_DIR])
    speed = weather[GFS_WIND_SPEED]
    out = weather.copy()
    out["wind_u_10m"] = -speed * np.sin(rad)
    out["wind_v_10m"] = -speed * np.cos(rad)
    return out


def build_processed(zone: str, data_root: Path | str = "data") -> pd.DataFrame:
    """Assemble the modeling-ready hourly frame for a zone (does not write)."""
    data_root = Path(data_root)
    ci = _load_endpoint(zone, "carbon-intensity/past-range", data_root)
    power = _load_endpoint(zone, "power-breakdown/past-range", data_root)
    flows = _load_endpoint(zone, "electricity-flows/past-range", data_root)
    weather = _derive_wind_uv(_load_weather(zone, data_root))

    prod_cols = [c for c in power.columns if c.startswith("prod_") and c.endswith("_mw")]
    prod = power[prod_cols]

    df = pd.DataFrame(index=ci.index)
    df["cons_based_ci"] = ci["carbon_intensity_gco2eq_kwh"]
    df["prod_based_ci_lifecycle"] = production_based_ci(prod, EF_LIFECYCLE)
    df["prod_based_ci_direct"] = production_based_ci(prod, EF_DIRECT)

    # Per-source production + derived generation aggregates.
    df = df.join(prod, how="left")
    total_gen = prod.clip(lower=0.0).sum(axis=1)
    df["total_generation_mw"] = total_gen
    renew_cols = [
        c for c in prod_cols
        if EM_SOURCE_TO_CC.get(c[len("prod_"):-len("_mw")]) in RENEWABLE_CC
    ]
    df["renewable_share"] = (
        prod[renew_cols].clip(lower=0.0).sum(axis=1) / total_gen.replace(0.0, np.nan)
    )

    # Flows: per-partner import/export plus net columns.
    df = df.join(flows, how="left")
    imports = flows[[c for c in flows.columns if c.startswith("import_")]].sum(axis=1)
    exports = flows[[c for c in flows.columns if c.startswith("export_")]].sum(axis=1)
    df["net_import_total_mw"] = imports - exports

    # Weather (incl. derived u/v) and calendar features.
    df = df.join(weather, how="left")
    df = df.join(calendar_features(df.index, zone), how="left")

    return df.sort_index()


def write_processed(zone: str, data_root: Path | str = "data") -> Path:
    """Build and atomically write data/processed/{zone}.parquet."""
    data_root = Path(data_root)
    df = build_processed(zone, data_root)
    path = storage.processed_path(zone, data_root)
    storage.write_parquet_atomic(df, path)
    logger.info("wrote %s: %d rows x %d cols", path, len(df), df.shape[1])
    return path
