"""Generate all raw datasets needed for thesis Chapter 6 case studies.

Output layout:
  data/samples/size_test/
      small_monthly_revenue.csv   ~120 rows  (6.2.4A – small)
      medium_daily_bakery.csv    ~1 000 rows  (6.2.4A – medium)
      large_hourly_factory.csv  ~10 000 rows  (6.2.4A – large)

  data/samples/join_discovery/
      air_passengers_base.csv     144 rows  (6.2.4B – base series, exact pool match)
      aviation_context.csv        144 rows  (6.2.4B – synthetic exog, same month key)
      product_catalog.csv          50 rows  (6.2.4B – no date → no-join trap)

  data/samples/broken/
      all_nan_target.csv           120 rows  (6.2.5 – all-NaN target)
      missing_target_col.csv        60 rows  (6.2.5 – 'sales' column absent)
      too_small_for_horizon.csv     15 rows  (6.2.5 – too few rows for horizon=12)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

rng = np.random.default_rng(42)

# ── Directories ───────────────────────────────────────────────────────────

for d in ["data/samples/size_test", "data/samples/join_discovery", "data/samples/broken"]:
    Path(d).mkdir(parents=True, exist_ok=True)

# ── 1. SMALL  (~120 rows) — monthly retail revenue  ──────────────────────
# 10 years of monthly data; trend + annual seasonality + noise.
months = pd.date_range("2014-01-01", "2023-12-01", freq="MS")
t = np.arange(len(months))
revenue = (
    50_000
    + 250 * t                                             # trend
    + 8_000 * np.sin(2 * np.pi * t / 12)                 # annual peak Dec
    + rng.normal(0, 1_500, len(months))
).clip(0).round(0).astype(int)

pd.DataFrame({
    "month": months.strftime("%Y-%m-%d"),
    "revenue_eur": revenue,
}).to_csv("data/samples/size_test/small_monthly_revenue.csv", index=False)
print(f"[size_test] small_monthly_revenue.csv  : {len(months):>6} rows")

# ── 2. MEDIUM (~1 000 rows) — daily bakery units sold  ───────────────────
days = pd.date_range("2021-01-01", periods=1_000, freq="D")
dow     = days.dayofweek.to_numpy()
trend_d = np.linspace(300, 420, len(days))
weekly  = 40 * np.sin(2 * np.pi * dow / 7 - np.pi / 2)
noise_d = rng.normal(0, 18, len(days))
units   = (trend_d + weekly + noise_d).clip(0).round(0).astype(int)

pd.DataFrame({
    "date": days.strftime("%Y-%m-%d"),
    "units_sold": units,
    "is_weekend": (dow >= 5).astype(int),
    "day_of_week": dow,
}).to_csv("data/samples/size_test/medium_daily_bakery.csv", index=False)
print(f"[size_test] medium_daily_bakery.csv    : {len(days):>6} rows")

# ── 3. LARGE (~10 000 rows) — hourly factory electricity  ────────────────
hours    = pd.date_range("2022-01-01", periods=10_000, freq="h")
h_arr    = hours.hour.to_numpy()
dow_h    = hours.dayofweek.to_numpy()
doy_h    = hours.dayofyear.to_numpy()
daily_c  = 400 * np.sin(2 * np.pi * h_arr / 24 - np.pi)
weekly_c = 150 * np.sin(2 * np.pi * dow_h / 7)
trend_h  = np.linspace(1_500, 1_750, len(hours))
noise_h  = rng.normal(0, 70, len(hours))
kwh      = (trend_h + daily_c + weekly_c + noise_h).clip(0).round(1)
temp_c   = (
    13 + 9 * np.sin(2 * np.pi * (doy_h - 90) / 365)
    + rng.normal(0, 2.5, len(hours))
).round(1)

pd.DataFrame({
    "timestamp": hours.strftime("%Y-%m-%d %H:%M:%S"),
    "consumption_kwh": kwh,
    "temperature_c": temp_c,
    "hour_of_day": h_arr,
    "is_weekend": (dow_h >= 5).astype(int),
}).to_csv("data/samples/size_test/large_hourly_factory.csv", index=False)
print(f"[size_test] large_hourly_factory.csv   : {len(hours):>6} rows")

# ── 4. JOIN DISCOVERY — air_passengers base (exact pool copy)  ───────────
data = sm.datasets.get_rdataset("AirPassengers", "datasets").data

def _float_year_to_date(y: float) -> str:
    year  = int(y)
    month = round((y - year) * 12) + 1
    return f"{year}-{month:02d}-01"

ap = pd.DataFrame({
    "month": data["time"].apply(_float_year_to_date),
    "passengers": data["value"].astype(int),
})
ap.to_csv("data/samples/join_discovery/air_passengers_base.csv", index=False)
print(f"[join_discovery] air_passengers_base.csv : {len(ap):>6} rows")

# ── 5. JOIN DISCOVERY — aviation context (synthetic exog, same month key)  ─
t_ap = np.arange(len(ap))
fuel_idx = (
    80
    + 12 * np.sin(2 * np.pi * t_ap / 12)       # annual cycle
    + 0.15 * t_ap                               # slow upward trend (oil)
    + rng.normal(0, 2.5, len(ap))
).round(1)
load_factor = (
    63
    + 9 * np.sin(2 * np.pi * t_ap / 12 + 0.4)  # slightly offset seasonal peak
    + rng.normal(0, 1.5, len(ap))
).clip(45, 95).round(1)

pd.DataFrame({
    "month": ap["month"],
    "fuel_price_idx": fuel_idx,
    "avg_load_factor_pct": load_factor,
}).to_csv("data/samples/join_discovery/aviation_context.csv", index=False)
print(f"[join_discovery] aviation_context.csv     : {len(ap):>6} rows")

# ── 6. JOIN DISCOVERY — product catalog (NO date → no-join trap)  ─────────
categories = ["Electronics", "Food", "Clothing", "Tools", "Sports"]
pd.DataFrame({
    "product_id":    [f"P{i:03d}" for i in range(1, 51)],
    "product_name":  [f"Product {i}" for i in range(1, 51)],
    "category":      rng.choice(categories, 50),
    "unit_price_eur": rng.uniform(2.99, 299.99, 50).round(2),
    "stock_units":   rng.integers(0, 500, 50),
    "supplier_code": [f"SUP-{rng.integers(10, 99)}" for _ in range(50)],
}).to_csv("data/samples/join_discovery/product_catalog.csv", index=False)
print("[join_discovery] product_catalog.csv       :     50 rows  (no-join trap)")

# ── 7. BROKEN — all-NaN target  ───────────────────────────────────────────
dates_nan = pd.date_range("2023-01-01", periods=120, freq="MS")
pd.DataFrame({
    "month":   dates_nan.strftime("%Y-%m-%d"),
    "revenue": [float("nan")] * 120,
}).to_csv("data/samples/broken/all_nan_target.csv", index=False)
print("[broken] all_nan_target.csv              :    120 rows  (revenue = all NaN)")

# ── 8. BROKEN — missing declared target column  ───────────────────────────
# The user is expected to declare target='sales', but the column is absent.
dates_miss = pd.date_range("2023-01-01", periods=60, freq="MS")
pd.DataFrame({
    "month":     dates_miss.strftime("%Y-%m-%d"),
    "feature_a": rng.normal(0, 1, 60).round(2),
    "feature_b": rng.integers(0, 10, 60),
    # 'sales' column intentionally absent
}).to_csv("data/samples/broken/missing_target_col.csv", index=False)
print("[broken] missing_target_col.csv          :     60 rows  ('sales' absent)")

# ── 9. BROKEN — too small for horizon  ────────────────────────────────────
# 15 rows; with horizon=12 there are only 3 training rows after val split.
pd.DataFrame({
    "month": pd.date_range("2023-01-01", periods=15, freq="MS").strftime("%Y-%m-%d"),
    "sales": rng.integers(100, 500, 15),
}).to_csv("data/samples/broken/too_small_for_horizon.csv", index=False)
print("[broken] too_small_for_horizon.csv       :     15 rows  (horizon=12 -> crash)")

print("\nDone. All datasets written.")
