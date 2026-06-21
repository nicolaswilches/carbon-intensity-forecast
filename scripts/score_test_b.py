"""Test B (live June 2026 window) scoring: 4-framework structural check on a
second held-out window, plus the head-to-head against Electricity Maps.

Two outputs:
  outputs/test_b_structural.csv   per zone x saved config, 1-24h accuracy over
                                  every hourly origin in the scored window.
  outputs/test_b_headtohead.csv   per zone, at EM snapshot origins, EM vs our
                                  consumption-based models on the 1-24h band.

Saved configs replayed (inference only, no retrain):
  single_prod  outputs/models_single_prod/{zone}   target prod_based_ci_lifecycle
  single_cons  outputs/models_single_cons/{zone}   target cons_based_ci
  e3_cons      outputs/models/{zone}                two-tier cons (source+flow)
(E2 two-tier prod is not saved locally and is omitted; it is the CarbonCast
reproduction already reported on Test A.)

Replay uses perfect (actual) future weather, matching the Test A protocol. The
window is restricted so each scored 24h horizon ends on settled actuals (EM
revises recent consumption-based CI for a few days), see SETTLE_CUT.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import keras
import numpy as np
import pandas as pd

from carbon_forecast.models.carboncast_extended import load_e3, predict_with_truth
from carbon_forecast.models.tier2_cnnlstm import Tier2Artifacts, assemble_sequences

ZONES = ["BE", "FI", "SG", "US-MIDA-PJM", "US-NY-NYIS"]
TZ = "UTC"
WINDOW_START = pd.Timestamp("2026-06-08", tz=TZ)   # live collection start
SETTLE_CUT = pd.Timestamp("2026-06-18 23:00", tz=TZ)  # last settled actual hour
SLICE_LO = "2026-05-25"          # generous lookback margin
SLICE_HI = "2026-06-21 23:00"    # weather actuals coverage end
H0, H1 = 1, 24                   # EM academic forecast horizon cap


def load_single(path: str | Path) -> Tier2Artifacts:
    p = Path(path)
    m = pickle.load(open(p / "meta.pkl", "rb"))
    model = keras.models.load_model(p / "tier2.keras", compile=False)
    return Tier2Artifacts(
        model=model, normalizer=m["normalizer"], target_col=m["target_col"],
        dynamic_cols=m["dynamic_cols"], weather_cols=m["weather_cols"],
        calendar_cols=m["calendar_cols"], config=m["config"],
    )


def single_predict_with_truth(art, frame):
    fn = art.normalizer.transform(frame)
    X, _, origins, _ = assemble_sequences(fn, art.config, 1, None)
    pn = art.model.predict(X, verbose=0)
    preds = np.clip(art.normalizer.inverse_transform(pn, art.target_col), 0.0, None)
    _, yn, _, _ = assemble_sequences(fn, art.config, 1, None)
    y_true = art.normalizer.inverse_transform(yn, art.target_col)
    return preds, y_true, origins


def band_metrics(preds, y_true, origins, origin_mask=None):
    """1-24h accuracy over origins in [WINDOW_START, SETTLE_CUT - 24h]."""
    keep = (origins >= WINDOW_START) & (origins + pd.Timedelta(hours=H1) <= SETTLE_CUT)
    if origin_mask is not None:
        keep = keep & origin_mask
    p = preds[keep][:, H0 - 1:H1]
    y = y_true[keep][:, H0 - 1:H1]
    err = p - y
    denom = np.clip(np.abs(y), 1e-6, None)
    return {
        "mape_pct": float(np.nanmean(np.abs(err) / denom) * 100),
        "mae": float(np.nanmean(np.abs(err))),
        "rmse": float(np.sqrt(np.nanmean(err ** 2))),
        "n_origins": int(keep.sum()),
    }


def em_snapshot_metrics(zone, cons_actual):
    """EM operational 1-24h accuracy at its own snapshot origins."""
    rows = []
    for fp in sorted(Path(f"data/raw/em/forecasts/{zone}").glob("*.parquet")):
        ef = pd.read_parquet(fp)
        origin = ef["snapshot_utc"].iloc[0].floor("h")
        if not (WINDOW_START <= origin and origin + pd.Timedelta(hours=H1) <= SETTLE_CUT):
            continue
        # forecast horizon 1..24 = origin+1 .. origin+24
        idx = pd.date_range(origin + pd.Timedelta(hours=1), periods=H1, freq="h", tz=TZ)
        fc = ef["carbon_intensity_gco2eq_kwh"].reindex(idx)
        ac = cons_actual.reindex(idx)
        m = fc.notna() & ac.notna()
        if m.sum() == 0:
            continue
        err = (fc[m] - ac[m]).to_numpy()
        denom = np.clip(np.abs(ac[m].to_numpy()), 1e-6, None)
        rows.append((origin, np.abs(err) / denom, np.abs(err), err))
    if not rows:
        return None, []
    origins = [r[0] for r in rows]
    ape = np.concatenate([r[1] for r in rows])
    ae = np.concatenate([r[2] for r in rows])
    e = np.concatenate([r[3] for r in rows])
    return {
        "mape_pct": float(np.mean(ape) * 100),
        "mae": float(np.mean(ae)),
        "rmse": float(np.sqrt(np.mean(e ** 2))),
        "n_origins": len(origins),
    }, origins


def main():
    struct_rows, h2h_rows = [], []
    for z in ZONES:
        frame = pd.read_parquet(f"data/processed/{z}.parquet")
        sub = frame.loc[SLICE_LO:SLICE_HI].copy()
        cons_actual = frame["cons_based_ci"]

        # --- structural: each saved config, all hourly origins ---
        configs = {
            "single_prod": ("single", f"outputs/models_single_prod/{z}"),
            "single_cons": ("single", f"outputs/models_single_cons/{z}"),
            "e3_cons":     ("e3",     f"outputs/models/{z}"),
        }
        per_config_pred = {}
        for name, (kind, path) in configs.items():
            if kind == "single":
                art = load_single(path)
                p, y, o = single_predict_with_truth(art, sub)
            else:
                art = load_e3(path)
                p, y, o = predict_with_truth(art, sub)
            per_config_pred[name] = (p, y, o)
            m = band_metrics(p, y, o)
            struct_rows.append({"zone": z, "config": name, **m})
            print(f"[struct] {z:14s} {name:11s} MAPE {m['mape_pct']:6.2f}  n={m['n_origins']}")

        # --- head-to-head: EM vs our cons models at EM snapshot origins ---
        em_m, em_origins = em_snapshot_metrics(z, cons_actual)
        if em_m is not None:
            h2h_rows.append({"zone": z, "model": "EM_operational", **em_m})
            print(f"[h2h]    {z:14s} {'EM_operational':16s} MAPE {em_m['mape_pct']:6.2f}  n={em_m['n_origins']}")
            em_set = pd.DatetimeIndex(em_origins)
            for name in ("single_cons", "e3_cons"):
                p, y, o = per_config_pred[name]
                mask = o.isin(em_set)
                m = band_metrics(p, y, o, origin_mask=mask)
                h2h_rows.append({"zone": z, "model": name, **m})
                print(f"[h2h]    {z:14s} {name:16s} MAPE {m['mape_pct']:6.2f}  n={m['n_origins']}")

    Path("outputs").mkdir(exist_ok=True)
    pd.DataFrame(struct_rows).to_csv("outputs/test_b_structural.csv", index=False)
    pd.DataFrame(h2h_rows).to_csv("outputs/test_b_headtohead.csv", index=False)
    print("\nwrote outputs/test_b_structural.csv and outputs/test_b_headtohead.csv")


if __name__ == "__main__":
    main()
