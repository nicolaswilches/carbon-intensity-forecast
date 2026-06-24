"""Assemble the final per-zone-window results: full history for the stable grids
(SG, NYIS, PJM), 2024-onward for the transitioning grids (FI, BE).

Produces a single source of truth that the figure and table scripts can read:

  outputs/final_metrics.csv          long table: zone, framework, window, metrics
  outputs/preds_final_<framework>/   one {zone}.npz per zone, picked per window

No retraining: the stable zones reuse the full-window artifacts already in the repo,
FI and BE use the validation-selected seed from the 2024-window run.

    .venv/bin/python scripts/make_final_results.py
"""
from __future__ import annotations

import os
import shutil

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "outputs")

FULL_ZONES = ["SG", "US-NY-NYIS", "US-MIDA-PJM"]
W2024_ZONES = ["FI", "BE"]
ZONES = ["SG", "US-NY-NYIS", "US-MIDA-PJM", "FI", "BE"]

# framework -> (full-window metrics CSV, full-window preds dir)
FULL = {
    "single_prod": ("single_tier_prod", "preds_single_prod"),
    "single_cons": ("single_tier_cons", "preds_single_cons"),
    "e2_prod": ("e2_realsplit_testA_valsel", "preds_e2_prod"),
    "e3_cons": ("e3_realsplit_testA_valsel", "preds"),
}
COLS = ["val_mape", "test_mape", "test_mae", "test_rmse"]


def main() -> None:
    new = pd.read_csv(os.path.join(OUT, "all_frameworks_train2024.csv")).set_index(["zone", "framework"])
    rows = []
    for fw, (csv, preds_dir) in FULL.items():
        full = pd.read_csv(os.path.join(OUT, f"{csv}.csv")).set_index("zone")
        dst = os.path.join(OUT, f"preds_final_{fw}")
        os.makedirs(dst, exist_ok=True)
        for z in ZONES:
            if z in FULL_ZONES:
                window, seed = "full", int(full.loc[z, "best_seed"])
                metrics = {c: float(full.loc[z, c]) for c in COLS}
                src = os.path.join(OUT, preds_dir, f"{z}.npz")
            else:
                r = new.loc[(z, fw)]
                window, seed = "2024", int(r["seed"])
                metrics = {c: float(r[c]) for c in COLS}
                src = os.path.join(OUT, "preds_train2024", f"{z}_{fw}_seed{seed}.npz")
            shutil.copyfile(src, os.path.join(dst, f"{z}.npz"))
            rows.append({"zone": z, "framework": fw, "window": window, "best_seed": seed, **metrics})

    final = pd.DataFrame(rows)[["zone", "framework", "window", "best_seed", *COLS]]
    final.to_csv(os.path.join(OUT, "final_metrics.csv"), index=False)

    piv = final.pivot(index="zone", columns="framework", values="test_mape").reindex(ZONES)
    win = final.pivot(index="zone", columns="framework", values="window").reindex(ZONES)
    print("=== final Test A MAPE (window in parentheses) ===")
    for z in ZONES:
        cells = " ".join(f"{fw}={piv.loc[z, fw]:.2f}({win.loc[z, fw][0]})"
                         for fw in FULL)
        print(f"  {z:12s} {cells}")
    print("\nwrote outputs/final_metrics.csv and preds_final_<framework>/ (5 npz each)")


if __name__ == "__main__":
    main()
