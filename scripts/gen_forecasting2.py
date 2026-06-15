"""Regenerate the forecasting2 example datasets as a 3-year (156-week) weekly series.

Produces four join-able CSVs (grid_demand, calendar_registry, meteorological,
generation_mix) sharing the same weekly dates but different date column names,
mirroring the original 1-year sample. Designed so the series is actually
learnable: a full 52-week yearly cycle repeated 3x, a mild upward load-growth
trend, and demand physically driven by temperature (heating + cooling), so
seasonal / trend / exogenous models can beat a naive constant.

Run:  uv run python scripts/gen_forecasting2.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OUT = "data/samples/forecasting2"
N_WEEKS = 156  # 3 years
START = "2023-01-02"  # Monday, matches original
rng = np.random.default_rng(42)

dates = pd.date_range(START, periods=N_WEEKS, freq="W-MON")
t = np.arange(N_WEEKS)
woy = t % 52  # week-of-year phase: 0 = early Jan (winter)
phase = 2 * np.pi * woy / 52

# --- Temperature: yearly cycle, cold in winter (woy 0), hot in summer (woy 26) ---
mean_temp = 9.0 - 13.0 * np.cos(phase) + rng.normal(0, 1.6, N_WEEKS)
mean_temp = mean_temp.round(1)

# --- Demand: base + load-growth trend + heating/cooling response to temp + noise ---
base = 2550.0
trend = (150.0 / 52.0) * t  # ~+150 MWh per year
heating = 25.0 * np.maximum(0.0, 15.0 - mean_temp)  # cold -> more demand
cooling = 20.0 * np.maximum(0.0, mean_temp - 20.0)  # hot  -> AC demand
demand = base + trend + heating + cooling + rng.normal(0, 55, N_WEEKS)
demand = demand.round(1)

# --- Coherent secondary grid columns ---
peak_demand = (demand * 0.20 + rng.normal(0, 12, N_WEEKS)).round(1)
grid_losses = (demand * 0.04 + rng.normal(0, 7, N_WEEKS)).round(1)

# --- Meteorological detail ---
min_temp = (mean_temp - rng.uniform(3, 6, N_WEEKS)).round(1)
max_temp = (mean_temp + rng.uniform(3, 6, N_WEEKS)).round(1)
wind = (5.0 + 2.5 * np.cos(phase) + rng.normal(0, 1.5, N_WEEKS)).clip(0.5).round(1)  # windier winter
humidity = (70.0 - 0.6 * mean_temp + rng.normal(0, 6, N_WEEKS)).clip(35, 99).round(1)

# --- Generation mix (renewables track wind, a touch up each year) ---
renewable = (32.0 + 1.5 * t / 52.0 + 3.0 * wind + rng.normal(0, 4, N_WEEKS)).clip(15, 75).round(1)
nuclear = (22.0 + rng.normal(0, 1.8, N_WEEKS)).round(1)
coal = (22.0 - 0.15 * (renewable - 32) + rng.normal(0, 2.5, N_WEEKS)).clip(8, 35).round(1)
imports = rng.integers(40, 350, N_WEEKS).astype(float)

# --- Calendar ---
months = dates.month
season_map = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring",
              5: "spring", 6: "summer", 7: "summer", 8: "summer", 9: "autumn",
              10: "autumn", 11: "autumn"}
season = [season_map[m] for m in months]
school_term = [0 if m in (7, 8) else 1 for m in months]  # summer break
# ~7 public-holiday weeks per year at fixed week-of-year positions
holiday_woys = {0, 14, 18, 30, 47, 51}
public_holidays = [(1 if w in holiday_woys else 0) for w in woy]
working_days = [5 - ph - (1 if w in (25, 26) else 0) for ph, w in zip(public_holidays, woy)]
working_days = [max(3, d) for d in working_days]
dst_change = [(1 if w in (12, 43) else 0) for w in woy]

ds = dates.strftime("%Y-%m-%d")

pd.DataFrame({
    "timestamp_week": ds, "demand_mwh": demand,
    "peak_demand_mw": peak_demand, "grid_losses_mwh": grid_losses,
}).to_csv(f"{OUT}/grid_demand.csv", index=False)

pd.DataFrame({
    "week_start_date": ds, "working_days": working_days,
    "public_holidays": public_holidays, "season": season,
    "school_term": school_term, "dst_change": dst_change,
}).to_csv(f"{OUT}/calendar_registry.csv", index=False)

pd.DataFrame({
    "obs_date": ds, "mean_temp_c": mean_temp, "min_temp_c": min_temp,
    "max_temp_c": max_temp, "wind_avg_ms": wind, "humidity_pct": humidity,
}).to_csv(f"{OUT}/meteorological.csv", index=False)

pd.DataFrame({
    "week_ref": ds, "renewable_share_pct": renewable,
    "nuclear_share_pct": nuclear, "coal_share_pct": coal, "import_mwh": imports,
}).to_csv(f"{OUT}/generation_mix.csv", index=False)

print(f"Wrote 4 files x {N_WEEKS} weeks ({ds[0]} -> {ds[-1]}) to {OUT}/")
print(f"demand_mwh  min/mean/max: {demand.min():.0f} / {demand.mean():.0f} / {demand.max():.0f}")
print(f"mean_temp_c min/mean/max: {mean_temp.min():.1f} / {mean_temp.mean():.1f} / {mean_temp.max():.1f}")
