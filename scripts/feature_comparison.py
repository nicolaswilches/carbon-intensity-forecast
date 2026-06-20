"""Feature comparison (Contribution 2): cross-border inputs on vs off.

Trains the consumption-based two-tier model (E3) with and without the cross-border
inputs and compares Test A accuracy per zone. "Flow off" is E3 with the net-flow
and partner-CI columns dropped, so the orchestrator builds a source-only Tier 2 (no
flow ANN, no partner-CI channel); "flow on" is the full E3. Each condition is run
over several seeds and the best-by-validation model is kept, matching the protocol
behind e3_realsplit_testA_valsel.csv (which is itself the flow-on baseline).

Colab-portable: set DATA_ROOT to the mounted data folder. The full config is a
multi-hour GPU job; pass FAST=1 for a quick, low-fidelity directional run.

    .venv/bin/python scripts/feature_comparison.py            # full
    FAST=1 ZONES=US-NY-NYIS SEEDS=0 .venv/bin/python scripts/feature_comparison.py
"""
from __future__ import annotations

import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from carbon_forecast.models.carboncast_extended import train_e3, evaluate_ci, E3Config  # noqa: E402
from carbon_forecast.models.tier1_source import Tier1Config  # noqa: E402
from carbon_forecast.models.tier1_flow import Tier1FlowConfig  # noqa: E402
from carbon_forecast.models.tier2_cnnlstm import Tier2Config  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(ROOT, "data", "processed"))
OUT = os.path.join(ROOT, "outputs")

ALL_ZONES = ["SG", "US-NY-NYIS", "US-MIDA-PJM", "FI", "BE"]
ZONES = os.environ.get("ZONES", ",".join(ALL_ZONES)).split(",")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
TRAIN, VAL, TEST = slice("2021", "2025"), slice("2026-01", "2026-04"), slice("2026-05", "2026-05")


def _cfg() -> E3Config:
    if os.environ.get("FAST"):
        t1 = Tier1Config(epochs=15, patience=4, stride=6)
        fl = Tier1FlowConfig(epochs=15, patience=4, stride=6)
        return E3Config(tier1=t1, tier1_fold=Tier1Config(epochs=10, stride=6),
                        flow=fl, flow_fold=Tier1FlowConfig(epochs=10, stride=6),
                        tier2=Tier2Config(epochs=20, patience=5, stride=2))
    return E3Config()  # full defaults (the multi-hour run)


def _drop_cross_border(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns
            if c.startswith("net_flow_") or c.startswith("partner_ci_")]
    return df.drop(columns=cols)


def run(zone: str, flow_on: bool, cfg: E3Config) -> dict:
    df = pd.read_parquet(os.path.join(DATA_ROOT, f"{zone}.parquet"))
    if not flow_on:
        df = _drop_cross_border(df)
    train, val, test = df.loc[TRAIN], df.loc[VAL], df.loc[TEST]
    runs = []
    for s in SEEDS:
        art = train_e3(train, val, zone, cfg, verbose=0, seed=s)
        v, t = evaluate_ci(art, val), evaluate_ci(art, test)
        runs.append((s, v["mape_pct"], t["mape_pct"], t["mae"], t["rmse"]))
        print(f"  {zone} flow={'on' if flow_on else 'off'} seed={s} "
              f"val={v['mape_pct']:.2f} test={t['mape_pct']:.2f}", flush=True)
    best = min(runs, key=lambda r: r[1])  # lowest validation MAPE
    return dict(zone=zone, flow="on" if flow_on else "off", best_seed=best[0],
                val_mape=round(best[1], 2), test_mape=round(best[2], 2),
                test_mae=round(best[3], 2), test_rmse=round(best[4], 2))


def main():
    cfg = _cfg()
    modes = [m == "on" for m in os.environ.get("MODES", "off,on").split(",")]
    rows = []
    for zone in ZONES:
        for flow_on in modes:
            t0 = time.time()
            row = run(zone, flow_on, cfg)
            row["minutes"] = round((time.time() - t0) / 60, 1)
            rows.append(row)
            print(f"{zone} flow={row['flow']}: test MAPE {row['test_mape']} "
                  f"({row['minutes']} min)", flush=True)
    out = os.path.join(OUT, "feature_comparison.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    main()
