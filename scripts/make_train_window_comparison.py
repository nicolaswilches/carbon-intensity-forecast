"""Compare full-window (2021-2025) vs shorter-window (2024-2025) training.

Joins the headline per-framework Test A metrics with the 2024-window run
(all_frameworks_train2024.csv) and writes a per zone/framework delta table.
Lower MAPE is better, so a negative delta means the shorter window helped.

    .venv/bin/python scripts/make_train_window_comparison.py
"""
from __future__ import annotations

import os

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "outputs")

# full-window CSV -> framework key used in the 2024 run
FULL = {
    "single_prod": "single_tier_prod",
    "single_cons": "single_tier_cons",
    "e2_prod": "e2_realsplit_testA_valsel",
    "e3_cons": "e3_realsplit_testA_valsel",
}
ZONES = ["SG", "US-NY-NYIS", "US-MIDA-PJM", "FI", "BE"]


def main() -> None:
    full_rows = []
    for fw, csv in FULL.items():
        df = pd.read_csv(os.path.join(OUT, f"{csv}.csv")).set_index("zone")
        for z in ZONES:
            full_rows.append({"zone": z, "framework": fw, "full_mape": float(df.loc[z, "test_mape"])})
    full = pd.DataFrame(full_rows)

    new = pd.read_csv(os.path.join(OUT, "all_frameworks_train2024.csv"))
    new = new[["zone", "framework", "test_mape"]].rename(columns={"test_mape": "w2024_mape"})

    m = full.merge(new, on=["zone", "framework"])
    m["delta"] = (m["w2024_mape"] - m["full_mape"]).round(2)
    m["pct_chg"] = ((m["delta"] / m["full_mape"]) * 100).round(1)
    m = m.sort_values(["zone", "framework"]).reset_index(drop=True)
    m.to_csv(os.path.join(OUT, "train_window_comparison.csv"), index=False)

    # pretty per-zone print
    for z in ZONES:
        print(f"\n{z}")
        sub = m[m.zone == z]
        for _, r in sub.iterrows():
            arrow = "better" if r.delta < 0 else ("worse" if r.delta > 0 else "same")
            print(f"  {r.framework:12s} full {r.full_mape:6.2f} -> 2024 {r.w2024_mape:6.2f}"
                  f"  ({r.delta:+.2f}, {r.pct_chg:+.1f}%) {arrow}")
    print("\nwrote outputs/train_window_comparison.csv")


if __name__ == "__main__":
    main()
