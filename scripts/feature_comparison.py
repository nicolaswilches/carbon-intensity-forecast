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

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from carbon_forecast.models.carboncast_extended import (  # noqa: E402
    train_e3, evaluate_ci, predict_with_truth, E3Config,
)
from carbon_forecast.models.tier1_source import Tier1Config  # noqa: E402
from carbon_forecast.models.tier1_flow import Tier1FlowConfig  # noqa: E402
from carbon_forecast.models.tier2_cnnlstm import Tier2Config  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(ROOT, "data", "processed"))
# OUTDIR lets a throwaway directional run write elsewhere, away from the real ledger.
OUT = os.environ.get("OUTDIR", os.path.join(ROOT, "outputs"))
PREDS_DIR = os.path.join(OUT, "preds_feature_comparison")  # one npz per zone/mode/seed

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


SEEDS_CSV = os.path.join(OUT, "feature_comparison_seeds.csv")


def _load_done() -> tuple[list[dict], set]:
    """Resume ledger: rows already computed in a previous (possibly killed) run."""
    if not os.path.exists(SEEDS_CSV):
        return [], set()
    df = pd.read_csv(SEEDS_CSV)
    return df.to_dict("records"), {(r.zone, r.flow, int(r.seed)) for r in df.itertuples()}


def run(zone: str, flow_on: bool, cfg: E3Config, done: set, ledger: list[dict]) -> list[dict]:
    """Train the not-yet-done seeds, save predictions + metrics, return all rows."""
    mode = "on" if flow_on else "off"
    df = pd.read_parquet(os.path.join(DATA_ROOT, f"{zone}.parquet"))
    if not flow_on:
        df = _drop_cross_border(df)
    train, val, test = df.loc[TRAIN], df.loc[VAL], df.loc[TEST]
    rows = [r for r in ledger if r["zone"] == zone and r["flow"] == mode]
    for s in SEEDS:
        if (zone, mode, s) in done:
            print(f"  {zone} flow={mode} seed={s} (cached, skip)", flush=True)
            continue
        art = train_e3(train, val, zone, cfg, verbose=0, seed=s)
        v, t = evaluate_ci(art, val), evaluate_ci(art, test)
        preds, y_true, origins = predict_with_truth(art, test)
        np.savez_compressed(os.path.join(PREDS_DIR, f"{zone}_{mode}_seed{s}.npz"),
                            preds=preds, y_true=y_true, origins=np.asarray(origins))
        row = dict(zone=zone, flow=mode, seed=s,
                   val_mape=round(v["mape_pct"], 2), test_mape=round(t["mape_pct"], 2),
                   test_mae=round(t["mae"], 2), test_rmse=round(t["rmse"], 2))
        rows.append(row)
        ledger.append(row)
        pd.DataFrame(ledger).to_csv(SEEDS_CSV, index=False)  # checkpoint per seed
        print(f"  {zone} flow={mode} seed={s} val={v['mape_pct']:.2f} "
              f"test={t['mape_pct']:.2f}", flush=True)
    return rows


def main():
    cfg = _cfg()
    modes = [m == "on" for m in os.environ.get("MODES", "off,on").split(",")]
    ledger, done = _load_done()
    for zone in ZONES:
        for flow_on in modes:
            t0 = time.time()
            rows = run(zone, flow_on, cfg, done, ledger)
            best = min(rows, key=lambda r: r["val_mape"])  # best-by-validation
            print(f"{zone} flow={'on' if flow_on else 'off'}: best-by-val test MAPE "
                  f"{best['test_mape']} ({round((time.time() - t0) / 60, 1)} min)", flush=True)

    # Summary: best-by-validation seed per (zone, mode).
    seeds = pd.DataFrame(ledger)
    summary = seeds.sort_values("val_mape").groupby(["zone", "flow"], as_index=False).first()
    summary.to_csv(os.path.join(OUT, "feature_comparison.csv"), index=False)
    print("wrote feature_comparison.csv (+ _seeds.csv, + preds_feature_comparison/)", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(PREDS_DIR, exist_ok=True)
    main()
