"""Regenerate the two Tier 1 diagnostic figures at single-column geometry.

Re-runs Tier 1 (per-source generation ANNs) on the real split (train 2021-2025,
val 2026 Jan-Apr, Test A May 2026), sums the per-source forecasts to a total-
generation forecast, and produces:

- tier1_horizon_error.pdf: normalized RMSE (RMSE / mean) of the total-generation
  forecast by horizon (1-96 h), per zone.
- tier1_pred_vs_actual_pjm.pdf: PJM day-ahead (24 h) gas-generation forecast vs
  actual over Test A.

Also prints the h1/h24/h48/h96 nRMSE per zone so the report's Tier 1 table can be
refreshed to match. Run with the project venv:

    .venv/bin/python scripts/make_tier1_figures.py
"""
from __future__ import annotations

import os
import sys

import keras
import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from carbon_forecast.models.tier1_source import (  # noqa: E402
    Tier1Config, train_source_model, predict_mw,
)
from carbon_forecast.plotting import config as P  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROC = os.path.join(ROOT, "data", "processed")
FIGS = os.path.join(ROOT, "outputs", "figures")
OUT = os.path.join(ROOT, "outputs")

ZONES = ["SG", "US-MIDA-PJM", "US-NY-NYIS", "FI", "BE"]
LABEL = {"BE": "Belgium", "FI": "Finland", "SG": "Singapore",
         "US-MIDA-PJM": "US-MIDA-PJM", "US-NY-NYIS": "US-NY-NYIS"}
CFG = Tier1Config(epochs=40, patience=6, stride=4)  # diagnostic budget, tractable on CPU
TRAIN, VAL, PRED = slice("2021", "2025"), slice("2026-01", "2026-04"), slice("2026-04", "2026-05")


def _load(z):
    return pd.read_parquet(os.path.join(PROC, f"{z}.parquet"))


def _sources(train):
    cols = [c for c in train.columns if c.startswith("prod_") and c.endswith("_mw")]
    tot = train[cols].sum(axis=1).mean()
    return [c[5:-3] for c in cols if train[c].mean() >= 0.01 * tot]


def tier1_total(z):
    """Return (horizons nRMSE %, gas preds-df) for one zone."""
    df = _load(z)
    train, val, pred = df.loc[TRAIN], df.loc[VAL], df.loc[PRED]
    keras.utils.set_random_seed(0)
    srcs = _sources(train)
    print(f"{z}: sources = {srcs}", flush=True)

    per_source: dict[str, pd.DataFrame] = {}
    gas = None
    for s in srcs:
        art = train_source_model(train, val, s, CFG, verbose=0)
        preds, origins = predict_mw(art, pred)
        d = pd.DataFrame(preds, index=pd.DatetimeIndex(origins))
        per_source[s] = d
        if z == "US-MIDA-PJM" and s == "gas":
            gas = d

    # Align on common origins, restrict to Test A (May 2026), sum to total.
    common = per_source[srcs[0]].index
    for s in srcs[1:]:
        common = common.intersection(per_source[s].index)
    common = common[(common.year == 2026) & (common.month == 5)]
    fcast = sum(per_source[s].loc[common].values for s in srcs)  # n x 96

    total = df[[f"prod_{s}_mw" for s in srcs]].sum(axis=1)
    pos = df.index.get_indexer(common)
    # preds[:, k] forecasts origin + k (k = 0..95); horizon h = k + 1.
    actual = np.stack([total.values[pos + k] for k in range(96)], axis=1)  # n x 96

    rmse_h = np.sqrt(np.mean((fcast - actual) ** 2, axis=0))
    nrmse = rmse_h / np.mean(actual, axis=0) * 100
    gas_out = None
    if gas is not None:
        g = gas.loc[common]
        gas_out = pd.DataFrame({  # day-ahead view = h24 (k=23)
            "t": common + pd.Timedelta(hours=23),
            "forecast": g.values[:, 23],
            "actual": df["prod_gas_mw"].values[pos + 23],
        })
    return nrmse, gas_out


def main():
    results, gas_pjm = {}, None
    for z in ZONES:
        nrmse, gas = tier1_total(z)
        results[z] = nrmse
        if gas is not None:
            gas_pjm = gas
        print(f"{z}: h1={nrmse[0]:.1f} h24={nrmse[23]:.1f} h48={nrmse[47]:.1f} "
              f"h96={nrmse[95]:.1f}", flush=True)

    pd.DataFrame({z: results[z] for z in ZONES}).to_csv(
        os.path.join(OUT, "tier1_horizon_nrmse.csv"), index_label="horizon")

    # Figure 12: per-zone nRMSE by horizon.
    fig = go.Figure()
    hours = np.arange(1, 97)
    for z in ZONES:
        fig.add_trace(go.Scatter(x=hours, y=results[z], mode="lines", name=LABEL[z],
                                 line=dict(color=P.REGIONAL_PALETTE[z], width=1.6)))
    P.style_report_fig(fig, span="column", height=340, legend=True,
                       xlabel="forecast horizon (hours ahead)",
                       ylabel="total-gen nRMSE (%)")
    fig.write_image(os.path.join(FIGS, "tier1_horizon_error.pdf"))
    print("wrote tier1_horizon_error.pdf", flush=True)

    # Figure 13: PJM day-ahead gas forecast vs actual.
    fig = go.Figure()
    t = gas_pjm["t"].dt.strftime("%Y-%m-%dT%H:%M:%S").tolist()
    fig.add_trace(go.Scatter(x=t, y=gas_pjm["actual"], mode="lines", name="actual",
                             line=dict(color="#7F7F7F", width=1.2)))
    fig.add_trace(go.Scatter(x=t, y=gas_pjm["forecast"], mode="lines",
                             name="forecast (24h-ahead)",
                             line=dict(color=P.REGIONAL_PALETTE["US-MIDA-PJM"], width=1.2)))
    P.style_report_fig(fig, span="column", height=300, legend=True, ylabel="gas generation (MW)")
    fig.write_image(os.path.join(FIGS, "tier1_pred_vs_actual_pjm.pdf"))
    print("wrote tier1_pred_vs_actual_pjm.pdf", flush=True)


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    main()
