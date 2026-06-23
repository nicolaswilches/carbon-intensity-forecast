"""Train and score all four frameworks on a configurable training window.

Frameworks per zone:
  single_prod    Tier 2 only, target prod_based_ci_lifecycle, no Tier 1 channels
  single_cons    Tier 2 only, target cons_based_ci
  e2_prod        CarbonCast-faithful two-tier, production target
  e3_cons        CarbonCast-extended two-tier, consumption target (flows + partner CI)

Each cell is one (zone, framework, seed). Seeds are run, the best-by-validation seed
is kept per (zone, framework), and every framework's model is saved (E2 included, via
save_e2), so all four frameworks have artifacts. Resumable via a ledger CSV.

The only thing that changes versus the headline run is the training window, set by
TRAIN_START (default 2024-01). Validation and Test A are unchanged. Outputs are
suffixed (SUFFIX, default train2024) so the full-window results stay side by side.

Colab-portable: set DATA_ROOT to the mounted data folder. Full config is a multi-hour
GPU job; pass FAST=1 for a quick low-fidelity smoke test.

    .venv/bin/python scripts/train_all_frameworks.py
    FAST=1 ZONES=SG SEEDS=0 FRAMEWORKS=single_cons .venv/bin/python scripts/train_all_frameworks.py
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path

import keras
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from carbon_forecast.models import tier2_cnnlstm as t2  # noqa: E402
from carbon_forecast.models.tier2_cnnlstm import (  # noqa: E402
    E2_TARGET, E3_TARGET, Tier2Config, assemble_sequences,
)
from carbon_forecast.models.tier1_source import Tier1Config  # noqa: E402
from carbon_forecast.models.tier1_flow import Tier1FlowConfig  # noqa: E402
from carbon_forecast.models.carboncast_faithful import (  # noqa: E402
    E2Config, train_e2, save_e2,
    evaluate_ci as e2_eval, predict_with_truth as e2_pwt,
)
from carbon_forecast.models.carboncast_extended import (  # noqa: E402
    E3Config, train_e3, save_e3,
    evaluate_ci as e3_eval, predict_with_truth as e3_pwt,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(ROOT, "data", "processed"))
OUT = os.environ.get("OUTDIR", os.path.join(ROOT, "outputs"))
SUFFIX = os.environ.get("SUFFIX", "train2024")
PREDS_DIR = os.path.join(OUT, f"preds_{SUFFIX}")
MODELS_DIR = os.path.join(OUT, f"models_{SUFFIX}")
SAVE_MODELS = os.environ.get("SAVE_MODELS", "1") != "0"
LEDGER_CSV = os.path.join(OUT, f"all_frameworks_{SUFFIX}_seeds.csv")
SUMMARY_CSV = os.path.join(OUT, f"all_frameworks_{SUFFIX}.csv")

ALL_ZONES = ["SG", "US-NY-NYIS", "US-MIDA-PJM", "FI", "BE"]
ZONES = os.environ.get("ZONES", ",".join(ALL_ZONES)).split(",")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1,2").split(",")]
ALL_FRAMEWORKS = ["single_prod", "single_cons", "e2_prod", "e3_cons"]
FRAMEWORKS = os.environ.get("FRAMEWORKS", ",".join(ALL_FRAMEWORKS)).split(",")

TRAIN_START = os.environ.get("TRAIN_START", "2024-01")
TRAIN = slice(TRAIN_START, "2025")
VAL = slice("2026-01", "2026-04")
TEST = slice("2026-05", "2026-05")
FAST = bool(os.environ.get("FAST"))


# --- single-tier helpers (Tier 2 standalone, no Tier 1 channels) ----------

def _single_cfg(target: str) -> Tier2Config:
    if FAST:
        return Tier2Config(target_col=target, dynamic_cols=[], epochs=20, patience=5, stride=2)
    return Tier2Config(target_col=target, dynamic_cols=[])


def _train_single(train, val, target, seed):
    keras.utils.set_random_seed(seed)
    return t2.train_tier2(train, val, _single_cfg(target), verbose=0)


def _single_pwt(art, frame):
    fn = art.normalizer.transform(frame)
    X, _, origins, _ = assemble_sequences(fn, art.config, 1, None)
    pn = art.model.predict(X, verbose=0)
    preds = np.clip(art.normalizer.inverse_transform(pn, art.target_col), 0.0, None)
    _, yn, _, _ = assemble_sequences(fn, art.config, 1, None)
    y = art.normalizer.inverse_transform(yn, art.target_col)
    return preds, y, origins


def _save_single(art, path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    art.model.save(p / "tier2.keras")
    meta = dict(normalizer=art.normalizer, config=art.config, target_col=art.target_col,
                dynamic_cols=art.dynamic_cols, weather_cols=art.weather_cols,
                calendar_cols=art.calendar_cols)
    with open(p / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)


def _e2_cfg() -> E2Config:
    if FAST:
        return E2Config(tier1=Tier1Config(epochs=15, patience=4, stride=6),
                        tier1_fold=Tier1Config(epochs=10, stride=6),
                        tier2=Tier2Config(target_col=E2_TARGET, epochs=20, patience=5, stride=2))
    return E2Config()


def _e3_cfg() -> E3Config:
    if FAST:
        return E3Config(tier1=Tier1Config(epochs=15, patience=4, stride=6),
                        tier1_fold=Tier1Config(epochs=10, stride=6),
                        flow=Tier1FlowConfig(epochs=15, patience=4, stride=6),
                        flow_fold=Tier1FlowConfig(epochs=10, stride=6),
                        tier2=Tier2Config(target_col=E3_TARGET, epochs=20, patience=5, stride=2))
    return E3Config()


# --- framework dispatch: train -> (val_metrics, test_metrics, preds), save ---

def _fit_eval(framework, train, val, test, zone, seed):
    if framework == "single_prod":
        art = _train_single(train, val, E2_TARGET, seed)
        return art, t2.evaluate_ci(art, val), t2.evaluate_ci(art, test), _single_pwt(art, test)
    if framework == "single_cons":
        art = _train_single(train, val, E3_TARGET, seed)
        return art, t2.evaluate_ci(art, val), t2.evaluate_ci(art, test), _single_pwt(art, test)
    if framework == "e2_prod":
        art = train_e2(train, val, zone, _e2_cfg(), verbose=0, seed=seed)
        return art, e2_eval(art, val), e2_eval(art, test), e2_pwt(art, test)
    if framework == "e3_cons":
        art = train_e3(train, val, zone, _e3_cfg(), verbose=0, seed=seed)
        return art, e3_eval(art, val), e3_eval(art, test), e3_pwt(art, test)
    raise ValueError(f"unknown framework {framework!r}")


def _save(framework, art, path):
    if framework.startswith("single"):
        _save_single(art, path)
    elif framework == "e2_prod":
        save_e2(art, path)
    elif framework == "e3_cons":
        save_e3(art, path)


def _load_done():
    if not os.path.exists(LEDGER_CSV):
        return [], set()
    df = pd.read_csv(LEDGER_CSV)
    return df.to_dict("records"), {(r.zone, r.framework, int(r.seed)) for r in df.itertuples()}


def run(zone, framework, df, done, ledger):
    train, val, test = df.loc[TRAIN], df.loc[VAL], df.loc[TEST]
    for s in SEEDS:
        if (zone, framework, s) in done:
            print(f"  {zone} {framework} seed={s} (cached, skip)", flush=True)
            continue
        art, v, t, (preds, y_true, origins) = _fit_eval(framework, train, val, test, zone, s)
        np.savez_compressed(os.path.join(PREDS_DIR, f"{zone}_{framework}_seed{s}.npz"),
                            preds=preds, y_true=y_true, origins=np.asarray(origins))
        if SAVE_MODELS:
            _save(framework, art, os.path.join(MODELS_DIR, framework, zone, f"seed{s}"))
        row = dict(zone=zone, framework=framework, seed=s, train_start=TRAIN_START,
                   val_mape=round(v["mape_pct"], 2), test_mape=round(t["mape_pct"], 2),
                   test_mae=round(t["mae"], 2), test_rmse=round(t["rmse"], 2))
        ledger.append(row)
        pd.DataFrame(ledger).to_csv(LEDGER_CSV, index=False)  # checkpoint per seed
        print(f"  {zone} {framework} seed={s} val={v['mape_pct']:.2f} "
              f"test={t['mape_pct']:.2f}", flush=True)


def main():
    ledger, done = _load_done()
    for zone in ZONES:
        df = pd.read_parquet(os.path.join(DATA_ROOT, f"{zone}.parquet"))
        for framework in FRAMEWORKS:
            t0 = time.time()
            run(zone, framework, df, done, ledger)
            print(f"{zone} {framework} done ({round((time.time() - t0) / 60, 1)} min)", flush=True)

    seeds = pd.DataFrame(ledger)
    if not seeds.empty:
        # best-by-validation seed per (zone, framework)
        best = seeds.sort_values("val_mape").groupby(["zone", "framework"], as_index=False).first()
        avg = (seeds.groupby(["zone", "framework"], as_index=False)["test_mape"]
               .mean().rename(columns={"test_mape": "test_mape_seedavg"}))
        summary = best.merge(avg, on=["zone", "framework"])
        summary.to_csv(SUMMARY_CSV, index=False)
        print(f"wrote {os.path.basename(SUMMARY_CSV)} (+ _seeds.csv, preds_{SUFFIX}/, models_{SUFFIX}/)",
              flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(PREDS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    main()
