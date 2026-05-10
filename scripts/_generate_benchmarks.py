"""Generate 6 small synthetic forecasting CSVs for the SP4 benchmark runner."""
import numpy as np
import pandas as pd
from pathlib import Path

Path("data/benchmarks").mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)

# 1. AirPassengers — univariate monthly, 144 obs
dates = pd.date_range("1949-01-01", periods=144, freq="MS")
trend = np.arange(144) * 1.8
seasonal = 60 * np.sin(np.arange(144) * 2 * np.pi / 12)
noise = rng.normal(scale=8, size=144)
pd.DataFrame({"month": dates, "passengers": np.round(112 + trend + seasonal + noise, 1)})\
  .to_csv("data/benchmarks/air_passengers.csv", index=False)

# 2. M4 monthly sample — 5 series × 120 obs
rows = []
for i, sid in enumerate(["M1", "M2", "M3", "M4", "M5"]):
    base = 200 + i * 50
    t = np.arange(120)
    y = base + t * 0.5 + 30 * np.sin(t * 2 * np.pi / 12) + rng.normal(scale=10, size=120)
    for d, v in zip(pd.date_range("2000-01-01", periods=120, freq="MS"), y):
        rows.append({"series_id": sid, "date": d, "value": round(v, 2)})
pd.DataFrame(rows).to_csv("data/benchmarks/m4_monthly_sample.csv", index=False)

# 3. Electricity demand — univariate daily, 730 obs
dates_d = pd.date_range("2022-01-01", periods=730, freq="D")
pd.DataFrame({
    "date": dates_d,
    "demand_gwh": np.round(
        400 + np.arange(730) * 0.05
        + 20 * np.sin(np.arange(730) * 2 * np.pi / 7)
        + 40 * np.sin(np.arange(730) * 2 * np.pi / 365)
        + rng.normal(scale=5, size=730), 2
    ),
}).to_csv("data/benchmarks/electricity_demand_sample.csv", index=False)

# 4. Sales — 3 stores × 156 weekly obs
rows = []
for sid in ["Store_A", "Store_B", "Store_C"]:
    base = int(rng.integers(500, 1500))
    t = np.arange(156)
    y = base + t * 0.3 + 100 * np.sin(t * 2 * np.pi / 52) + rng.normal(scale=30, size=156)
    for d, v in zip(pd.date_range("2020-01-01", periods=156, freq="W"), y):
        rows.append({"store_id": sid, "week": d, "sales": max(0, round(v, 0))})
pd.DataFrame(rows).to_csv("data/benchmarks/sales_sample.csv", index=False)

# 5. Weather — univariate daily temperature, 365 obs
dates_w = pd.date_range("2023-01-01", periods=365, freq="D")
temp = 15 + 12 * np.sin(np.arange(365) * 2 * np.pi / 365 - np.pi / 2) + rng.normal(scale=3, size=365)
pd.DataFrame({"date": dates_w, "temp_c": np.round(temp, 1)}).to_csv(
    "data/benchmarks/weather_sample.csv", index=False
)

# 6. Stock — 2 tickers × 252 daily obs
rows = []
for ticker in ["STOCK_A", "STOCK_B"]:
    price = 100.0
    prices = []
    for _ in range(252):
        price *= np.exp(rng.normal(0.0003, 0.015))
        prices.append(round(price, 2))
    for d, p in zip(pd.date_range("2023-01-01", periods=252, freq="B"), prices):
        rows.append({"ticker": ticker, "date": d, "close": p})
pd.DataFrame(rows).to_csv("data/benchmarks/stock_sample.csv", index=False)

print("Generated 6 benchmark CSVs in data/benchmarks/")
