"""Time-series (panel) version of the partial-coverage join stress test.

A multi-series weekly sales panel (one series per store) is joined to two store
dimension tables on a NON-date FK (store_id) with partial coverage:
  - store_directory : ORPHAN FKs (panel references stores absent from the directory)
  - store_demographics : FK SUBSET OF PK (directory has extra, unused stores)

Run: uv run python scripts/gen_partial_join_timeseries.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(11)
OUT = Path("data/samples/join_partial_coverage_ts")
OUT.mkdir(parents=True, exist_ok=True)

N_STORES = 20
N_WEEKS = 60
weeks = pd.date_range("2022-01-03", periods=N_WEEKS, freq="W-MON")
stores = [f"ST{i:02d}" for i in range(1, N_STORES + 1)]

# ── Base panel: one weekly series per store ─────────────────────────────────
rows = []
for s_idx, store in enumerate(stores):
    level = 200 + s_idx * 25                      # store-specific base level
    trend = rng.uniform(-0.5, 1.5)               # weekly trend
    amp = rng.uniform(20, 60)                     # seasonal amplitude
    for w_idx, wk in enumerate(weeks):
        seasonal = amp * np.sin(2 * np.pi * w_idx / 52.0)   # annual cycle
        noise = rng.normal(0, 12)
        units = max(0, level + trend * w_idx + seasonal + noise)
        rows.append({"week": wk.strftime("%Y-%m-%d"), "store_id": store,
                     "units_sold": int(round(units))})
base = pd.DataFrame(rows)
base.to_csv(OUT / "store_weekly_sales.csv", index=False)

# ── Dim 1: store_directory — ORPHAN FKs (left_coverage < 0.80) ──────────────
# Directory only has ST01..ST15; panel references ST01..ST20 -> 5 orphans.
# Key name differs (store_code vs store_id). Includes a duplicate PK row.
directory = pd.DataFrame({
    "store_code": [f"ST{i:02d}" for i in range(1, 16)],
    "region": rng.choice(["north", "south", "east", "west"], 15),
    "store_size_sqm": rng.integers(200, 2000, 15),
    "store_type": rng.choice(["flagship", "standard", "express"], 15),
})
dup = directory[directory["store_code"] == "ST05"].copy()
dup["region"] = "east"
directory = pd.concat([directory, dup], ignore_index=True)
directory.to_csv(OUT / "store_directory.csv", index=False)

# ── Dim 2: store_demographics — FK SUBSET OF PK (clean, numeric exog) ────────
# Demographics has ST01..ST30 (extra unused stores); panel uses ST01..ST20.
# Key name differs (store_ref). Numeric attributes -> safe as exogenous.
demo = pd.DataFrame({
    "store_ref": [f"ST{i:02d}" for i in range(1, 31)],
    "population_k": rng.integers(20, 500, 30),
    "median_age": rng.uniform(28, 52, 30).round(1),
    "urban_flag": rng.integers(0, 2, 30),
})
demo.to_csv(OUT / "store_demographics.csv", index=False)

print("Wrote:")
for p in sorted(OUT.glob("*.csv")):
    df = pd.read_csv(p)
    print(f"  {p.name:28} {len(df):5} rows, cols={list(df.columns)}")
