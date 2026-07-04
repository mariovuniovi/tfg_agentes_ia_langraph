"""Single-series time-series partial-coverage join stress test.

The forecasting analogue of the regression partial-coverage case (Sec. 6.2.6),
but trainable in V1 (single series, no series_id). One weekly demand series is
joined on the DATE key (names differ) to two context tables:
  - weather_weekly  : PARTIAL coverage (~75% of weeks present) -> orphan weeks
                      whose exogenous values are imputed (unknown_future obs).
  - calendar_weekly : CLEAN, FK SUBSET OF PK (extra future weeks), known_future.

Run: uv run python scripts/gen_partial_join_ts_single.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
OUT = Path("data/samples/join_partial_coverage_ts_single")
OUT.mkdir(parents=True, exist_ok=True)

N = 156  # ~3 years of weekly data, single series
weeks = pd.date_range("2021-01-04", periods=N, freq="W-MON")
t = np.arange(N)
woy = weeks.isocalendar().week.to_numpy()

# Seasonal weather + holiday signal drive demand (so the exog is informative).
temp = 12 + 10 * np.sin(2 * np.pi * (t % 52) / 52 - np.pi / 2) + rng.normal(0, 2, N)
holiday = (((woy >= 51) | (woy <= 1)) | ((woy >= 26) & (woy <= 31))).astype(int)
demand = (
    500 + 1.2 * t + 60 * np.sin(2 * np.pi * (t % 52) / 52)
    + 3 * (temp - 12) + 40 * holiday + rng.normal(0, 18, N)
).clip(0).round().astype(int)

# ── Base: single weekly series ──────────────────────────────────────────────
pd.DataFrame({"week": weeks.strftime("%Y-%m-%d"), "units_sold": demand}).to_csv(
    OUT / "weekly_demand.csv", index=False
)

# ── Context 1: weather — PARTIAL coverage (~75%), orphan weeks, key 'week_ref' ─
mask = rng.random(N) < 0.75
pd.DataFrame({
    "week_ref": weeks.strftime("%Y-%m-%d")[mask],
    "mean_temp_c": temp[mask].round(1),
    "rainfall_mm": rng.gamma(2.0, 8.0, int(mask.sum())).round(1),
}).to_csv(OUT / "weather_weekly.csv", index=False)

# ── Context 2: calendar — CLEAN, FK subset of PK (extra future weeks) ─────────
future = pd.date_range(weeks[-1] + pd.Timedelta(weeks=1), periods=12, freq="W-MON")
allw = weeks.append(future)
woy_all = allw.isocalendar().week.to_numpy()
hol_all = (((woy_all >= 51) | (woy_all <= 1)) | ((woy_all >= 26) & (woy_all <= 31))).astype(int)
pd.DataFrame({
    "week_key": allw.strftime("%Y-%m-%d"),
    "is_holiday_week": hol_all,
    "week_of_year": woy_all,
}).to_csv(OUT / "calendar_weekly.csv", index=False)

# ── Posted schema (the joined-target schema the UI would post) ────────────────
schema = {
    "problem_type": "forecasting",
    "name": "weekly_demand_partial_join",
    "description": (
        "Single weekly demand series (~156 weeks) joined on the date key to a "
        "partial-coverage weather table (~75% of weeks; orphan weeks imputed) and "
        "a clean calendar table. Forecasting analogue of the regression "
        "partial-coverage case: trainable single series with exogenous joins."
    ),
    "target_column": "units_sold",
    "datetime_column": "week",
    "forecast_horizon": 8,
    "frequency": "W-MON",
    "exogenous_columns": [
        {"name": "mean_temp_c", "future_availability": "unknown_future", "temporal": True},
        {"name": "rainfall_mm", "future_availability": "unknown_future", "temporal": True},
        {"name": "is_holiday_week", "future_availability": "known_future"},
        {"name": "week_of_year", "future_availability": "known_future"},
    ],
    "columns": [
        {"name": "week", "dtype": "datetime", "required": True, "nullable": False, "unique": True},
        {"name": "units_sold", "dtype": "int", "required": True, "nullable": False, "min": 0},
    ],
}
(OUT / "weekly_demand_schema.json").write_text(
    json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8"
)

cov = float(mask.mean())
print(f"[ts-single] weekly_demand.csv {N}x2  weather coverage={cov:.1%} "
      f"(orphans={N - int(mask.sum())})  calendar={len(allw)} weeks (clean)")
print(f"[ts-single] wrote 3 CSVs + schema to {OUT}")
