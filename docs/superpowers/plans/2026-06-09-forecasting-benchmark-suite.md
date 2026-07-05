# Curated Forecasting Benchmark Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Seed the experience pool with 14 reproducible, regime-balanced forecasting datasets (statistical / supervised / random-walk) so the planner's retrieval has quality priors.

**Architecture:** Extend the existing `scripts/_dataset_sources.py::fetch_dataset()` with two new source types (`statsmodels`, `uci_url`); add 14 curated entries to `benchmark_manifest.yaml` using those sources; add `--reset-forecasting` + family sanity-check to `run_benchmark.py`; add `reset_forecasting_experiences()` to `ExperiencePool`. No new scripts — the existing benchmark runner handles fetch → stage → train → pool-insert as before. `statsmodels` is added as a dependency.

**Tech Stack:** pandas, statsmodels (new dep), yfinance (existing), scikit-learn fetch_openml (existing), PyYAML, SQLite

---

## File Structure

| File | Action | Purpose |
|---|---|---|
| `pyproject.toml` | Modify | Add `statsmodels>=0.14` |
| `scripts/_dataset_sources.py` | Modify | Add `statsmodels` + `uci_url` source handlers |
| `scripts/benchmark_manifest.yaml` | Modify | Replace 7 forecasting entries with 14 curated entries |
| `scripts/run_benchmark.py` | Modify | `--reset-forecasting`, family sanity-check, `--strict` |
| `src/mlops_agents/experience/pool.py` | Modify | Add `reset_forecasting_experiences() -> int` |
| `tests/test_scripts/test_benchmark_datasets.py` | Create | Unit tests for new source handlers + family-check |

---

### Task 1: statsmodels dep + new source handlers in _dataset_sources.py

Adds `source: statsmodels` (for the 4 statistical datasets) and `source: uci_url` (for the 5 supervised datasets) to the existing `fetch_dataset()` dispatcher. Each handler returns a ready-to-use `pd.DataFrame` in the same shape expected by `stage_dataset()`.

**Files:**
- Modify: `pyproject.toml`
- Modify: `scripts/_dataset_sources.py`

- [ ] **Step 1: Write the failing unit tests first**

Create `tests/test_scripts/test_benchmark_datasets.py`:

```python
"""Unit tests for new _dataset_sources handlers.

All tests are offline — they mock HTTP and use only statsmodels (static data).
"""
from __future__ import annotations
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# statsmodels source handlers
# ---------------------------------------------------------------------------

def test_fetch_statsmodels_co2():
    from scripts._dataset_sources import fetch_dataset
    entry = {"source": "statsmodels", "source_id": "co2", "dataset_id": "co2_mauna_loa"}
    df = fetch_dataset(entry)
    assert "date" in df.columns
    assert "co2" in df.columns
    assert len(df) >= 250
    # monthly dates — all first-of-month
    assert pd.to_datetime(df["date"]).dt.day.eq(1).all()


def test_fetch_statsmodels_sunspots():
    from scripts._dataset_sources import fetch_dataset
    entry = {"source": "statsmodels", "source_id": "sunspots", "dataset_id": "sunspots"}
    df = fetch_dataset(entry)
    assert "year" in df.columns
    assert "sunspots" in df.columns
    assert len(df) >= 200


def test_fetch_statsmodels_nile():
    from scripts._dataset_sources import fetch_dataset
    entry = {"source": "statsmodels", "source_id": "nile", "dataset_id": "nile"}
    df = fetch_dataset(entry)
    assert "year" in df.columns
    assert "volume" in df.columns
    assert len(df) >= 90


def test_fetch_statsmodels_air_passengers():
    from scripts._dataset_sources import fetch_dataset
    entry = {"source": "statsmodels", "source_id": "air_passengers",
             "dataset_id": "air_passengers"}
    df = fetch_dataset(entry)
    assert "month" in df.columns
    assert "passengers" in df.columns
    assert len(df) == 144


def test_fetch_statsmodels_unknown_raises():
    from scripts._dataset_sources import fetch_dataset
    with pytest.raises(ValueError, match="Unknown statsmodels"):
        fetch_dataset({"source": "statsmodels", "source_id": "nope", "dataset_id": "nope"})


# ---------------------------------------------------------------------------
# uci_url handler — mock HTTP to avoid network in unit tests
# ---------------------------------------------------------------------------

def _make_metro_raw() -> str:
    """Minimal Metro Traffic Volume CSV content."""
    rows = []
    rows.append("holiday,temp,rain_1h,snow_1h,clouds_all,weather_main,"
                "weather_description,date_time,traffic_volume")
    for i in range(50):
        dt = pd.Timestamp("2018-01-01") + pd.Timedelta(hours=i)
        holiday = "None" if i % 24 != 0 else "New Years Day"
        rows.append(f"{holiday},280.0,0.0,0.0,40,Clouds,broken clouds,"
                    f"{dt.strftime('%Y-%m-%d %H:%M:%S')},3000")
    import io
    return "\n".join(rows)


def test_fetch_uci_url_metro(monkeypatch):
    import io
    import pandas as pd
    from scripts._dataset_sources import fetch_dataset

    raw_csv = _make_metro_raw()

    def _mock_read_csv(url, **kwargs):
        kwargs.pop("parse_dates", None)
        df = pd.read_csv(io.StringIO(raw_csv))
        if "parse_dates" in kwargs or True:
            df["date_time"] = pd.to_datetime(df["date_time"])
        return df

    monkeypatch.setattr("pandas.read_csv", _mock_read_csv)

    url = "https://example.com/metro.csv.gz"
    entry = {"source": "uci_url", "source_id": url,
             "dataset_id": "metro_traffic_volume"}
    df = fetch_dataset(entry)

    assert "date" in df.columns
    assert "traffic_volume" in df.columns
    assert "is_holiday" in df.columns
    assert len(df) >= 1  # at least one daily row
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest tests/test_scripts/test_benchmark_datasets.py -v
```
Expected: `ImportError` for missing `statsmodels` and `ModuleNotFoundError` / handler-not-found errors.

- [ ] **Step 3: Add statsmodels to pyproject.toml**

In `pyproject.toml`, inside `dependencies`, add after `"yfinance>=0.2"`:
```toml
    "statsmodels>=0.14",
```

Run:
```bash
uv sync
```
Expected: resolves without error.

- [ ] **Step 4: Add source handlers to scripts/_dataset_sources.py**

At the end of `fetch_dataset()`, just before the final `raise ValueError`, add the two new source branches:

```python
    if src == "statsmodels":
        return _fetch_statsmodels(entry)

    if src == "uci_url":
        return _fetch_uci_url(entry)
```

Then add the two helper functions after `fetch_dataset`:

```python
# ---------------------------------------------------------------------------
# statsmodels source — static public datasets (co2, sunspots, nile,
# air_passengers). No network after statsmodels is installed.
# ---------------------------------------------------------------------------

def _fetch_statsmodels(entry: dict) -> pd.DataFrame:
    import statsmodels.api as sm
    sid = entry["source_id"]

    if sid == "co2":
        raw = sm.datasets.co2.load_pandas().data  # weekly, DatetimeIndex, col 'co2'
        monthly = raw.resample("MS").mean().interpolate("linear").dropna()
        df = monthly.reset_index()
        df.columns = ["date", "co2"]
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        return df

    if sid == "sunspots":
        raw = sm.datasets.sunspots.load_pandas().data.copy()
        raw.columns = [c.upper() for c in raw.columns]  # normalise capitalisation
        return pd.DataFrame({
            "year": raw["YEAR"].astype(int).apply(lambda y: f"{y}-01-01"),
            "sunspots": raw["SUNACTIVITY"].values,
        })

    if sid == "nile":
        raw = sm.datasets.nile.load_pandas().data.reset_index()
        raw.columns = [c.lower() for c in raw.columns]
        return pd.DataFrame({
            "year": raw["year"].astype(int).apply(lambda y: f"{y}-01-01"),
            "volume": raw["volume"].values,
        })

    if sid == "air_passengers":
        data = sm.datasets.get_rdataset("AirPassengers", "datasets").data
        # 'time' is a float year (1949.0, 1949.0833, ...); 'value' is the count
        def _fy(y: float) -> str:
            year = int(y)
            month = round((y - year) * 12) + 1
            return f"{year}-{month:02d}-01"
        return pd.DataFrame({
            "month": data["time"].apply(_fy),
            "passengers": data["value"].astype(int),
        })

    raise ValueError(f"Unknown statsmodels dataset source_id: {sid!r}. "
                     f"Valid: co2, sunspots, nile, air_passengers")


# ---------------------------------------------------------------------------
# uci_url source — downloads a CSV from a UCI archive URL and applies
# dataset-specific resampling. Identified by entry['dataset_id'].
# ---------------------------------------------------------------------------

def _fetch_uci_url(entry: dict) -> pd.DataFrame:
    dataset_id = entry["dataset_id"]
    url = entry["source_id"]

    if dataset_id == "metro_traffic_volume":
        # UCI 492 — hourly Metro Interstate Traffic Volume → resample to daily
        df = pd.read_csv(url, parse_dates=["date_time"])
        df = df.set_index("date_time")
        df["is_holiday"] = (df["holiday"] != "None").astype(int)
        agg = {"traffic_volume": "mean", "temp": "mean", "rain_1h": "mean",
               "snow_1h": "mean", "clouds_all": "mean", "is_holiday": "max"}
        daily = df[list(agg)].resample("D").agg(agg).dropna().reset_index()
        daily = daily.rename(columns={"date_time": "date"})
        daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")
        return daily

    if dataset_id == "beijing_pm25":
        # UCI 381 — hourly Beijing PM2.5 → resample to daily
        df = pd.read_csv(url)
        df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
        keep = ["pm2.5", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"]
        daily = df.set_index("datetime")[keep].resample("D").mean().dropna().reset_index()
        daily = daily.rename(columns={"datetime": "date", "pm2.5": "pm25"})
        daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")
        return daily

    if dataset_id == "appliances_energy":
        # UCI 374 — 10-min Appliances Energy → resample to hourly (~3,300 rows)
        df = pd.read_csv(url, parse_dates=["date"])
        df = df.set_index("date")
        keep = ["Appliances", "T6", "RH_6", "T_out", "RH_out",
                "Windspeed", "Visibility", "Press_mm_hg"]
        hourly = df[keep].resample("h").mean().dropna().reset_index()
        hourly["date"] = hourly["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
        return hourly

    if dataset_id == "vic_elec":
        # Half-hourly Victorian electricity demand → resample to daily
        # Source: tsibbledata R package (GitHub raw).
        df = pd.read_csv(url, parse_dates=["Time"])
        df = df.rename(columns={"Time": "datetime", "Demand": "demand",
                                 "Temperature": "temperature", "Holiday": "holiday"})
        df["holiday"] = df["holiday"].astype(int)
        daily = df.set_index("datetime")[["demand", "temperature", "holiday"]].resample("D").agg(
            {"demand": "sum", "temperature": "mean", "holiday": "max"}
        ).dropna().reset_index()
        daily = daily.rename(columns={"datetime": "date"})
        daily["date"] = daily["date"].dt.strftime("%Y-%m-%d")
        return daily

    raise ValueError(f"No uci_url handler for dataset_id: {dataset_id!r}. "
                     f"Valid: metro_traffic_volume, beijing_pm25, "
                     f"appliances_energy, vic_elec")
```

- [ ] **Step 5: Run tests — should now pass**

```bash
uv run pytest tests/test_scripts/test_benchmark_datasets.py -v
```
Expected: all tests pass (statsmodels tests hit real data; metro test uses monkeypatched pd.read_csv).

- [ ] **Step 6: Run full suite to confirm no regressions**

```bash
uv run pytest -m "not integration" -q
```
Expected: same pass count as before + new tests green.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml scripts/_dataset_sources.py \
        tests/test_scripts/__init__.py \
        tests/test_scripts/test_benchmark_datasets.py
git commit -m "feat(benchmark): add statsmodels + uci_url source handlers to fetch_dataset"
```

---

### Task 2: Replace forecasting section in benchmark_manifest.yaml

Replaces all 7 stale forecasting entries with 14 curated entries. Statistical entries use `source: statsmodels`; supervised entries use `source: openml` (bike_sharing_daily) or `source: uci_url` (others); financial entries reuse `source: yfinance_multi` with existing fixed end-dates (sp500, oil, gold) or `source: local` for already-committed CSVs (crypto, fx). Every forecasting entry has `expected_family`.

**Files:**
- Modify: `scripts/benchmark_manifest.yaml`

- [ ] **Step 1: Delete the existing forecasting section and replace it**

In `scripts/benchmark_manifest.yaml`, delete everything from the `# Forecasting` comment (line ~94) to the end of the file, and replace with:

```yaml
# ---------------------------------------------------------------------------
# Forecasting (14) — curated, regime-balanced
# 3 regimes: statistical, supervised, random_walk
# ---------------------------------------------------------------------------

# ── Statistical (4): short/univariate → ETS / AutoARIMA wins ────────────────

- dataset_id: air_passengers
  source: statsmodels
  source_id: air_passengers
  problem_type: forecasting
  target_column: passengers
  datetime_column: month
  series_id_columns: []
  frequency: MS
  forecast_horizon: 12
  expected_family: statistical

- dataset_id: co2_mauna_loa
  source: statsmodels
  source_id: co2
  problem_type: forecasting
  target_column: co2
  datetime_column: date
  series_id_columns: []
  frequency: MS
  forecast_horizon: 6
  expected_family: statistical

- dataset_id: sunspots
  source: statsmodels
  source_id: sunspots
  problem_type: forecasting
  target_column: sunspots
  datetime_column: year
  series_id_columns: []
  frequency: YS
  forecast_horizon: 3
  expected_family: statistical

- dataset_id: nile
  source: statsmodels
  source_id: nile
  problem_type: forecasting
  target_column: volume
  datetime_column: year
  series_id_columns: []
  frequency: YS
  forecast_horizon: 3
  expected_family: statistical

# ── Supervised (5): long history + exog → tree ensembles win ────────────────

# OpenML 42712 — UCI Bike Sharing daily (dteday, cnt + weather/calendar exog)
- dataset_id: bike_sharing_daily
  source: openml
  source_id: 42712
  problem_type: forecasting
  target_column: cnt
  datetime_column: dteday
  series_id_columns: []
  frequency: D
  forecast_horizon: 14
  expected_family: supervised
  exogenous_columns:
    - { name: temp,       future_availability: unknown_future }
    - { name: hum,        future_availability: unknown_future }
    - { name: windspeed,  future_availability: unknown_future }
    - { name: season,     future_availability: known_future }
    - { name: holiday,    future_availability: known_future }
    - { name: workingday, future_availability: known_future }
    - { name: weekday,    future_availability: known_future }

# UCI 492 — Metro Interstate Traffic Volume hourly → daily
- dataset_id: metro_traffic_volume
  source: uci_url
  source_id: "https://archive.ics.uci.edu/ml/machine-learning-databases/00492/Metro_Interstate_Traffic_Volume.csv.gz"
  problem_type: forecasting
  target_column: traffic_volume
  datetime_column: date
  series_id_columns: []
  frequency: D
  forecast_horizon: 14
  expected_family: supervised
  exogenous_columns:
    - { name: temp,       future_availability: unknown_future }
    - { name: rain_1h,    future_availability: unknown_future }
    - { name: snow_1h,    future_availability: unknown_future }
    - { name: clouds_all, future_availability: unknown_future }
    - { name: is_holiday, future_availability: known_future }

# UCI 381 — Beijing PM2.5 hourly 2010-2014 → daily
- dataset_id: beijing_pm25
  source: uci_url
  source_id: "https://archive.ics.uci.edu/ml/machine-learning-databases/00381/PRSA_data_2010.1.1-2014.12.31.csv"
  problem_type: forecasting
  target_column: pm25
  datetime_column: date
  series_id_columns: []
  frequency: D
  forecast_horizon: 14
  expected_family: supervised
  exogenous_columns:
    - { name: DEWP, future_availability: unknown_future }
    - { name: TEMP, future_availability: unknown_future }
    - { name: PRES, future_availability: unknown_future }
    - { name: Iws,  future_availability: unknown_future }
    - { name: Is,   future_availability: unknown_future }
    - { name: Ir,   future_availability: unknown_future }

# UCI 374 — Appliances Energy 10-min → hourly (~3,300 rows; lightgbm/xgboost eligible)
- dataset_id: appliances_energy
  source: uci_url
  source_id: "https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv"
  problem_type: forecasting
  target_column: Appliances
  datetime_column: date
  series_id_columns: []
  frequency: H
  forecast_horizon: 24
  expected_family: supervised
  exogenous_columns:
    - { name: T6,          future_availability: unknown_future }
    - { name: RH_6,        future_availability: unknown_future }
    - { name: T_out,       future_availability: unknown_future }
    - { name: RH_out,      future_availability: unknown_future }
    - { name: Windspeed,   future_availability: unknown_future }
    - { name: Visibility,  future_availability: unknown_future }
    - { name: Press_mm_hg, future_availability: unknown_future }

# Victorian electricity demand 30-min → daily
- dataset_id: vic_elec
  source: uci_url
  source_id: "https://raw.githubusercontent.com/tidyverts/tsibbledata/master/data-raw/vic_elec/vic_elec.csv"
  problem_type: forecasting
  target_column: demand
  datetime_column: date
  series_id_columns: []
  frequency: D
  forecast_horizon: 14
  expected_family: supervised
  exogenous_columns:
    - { name: temperature, future_availability: unknown_future }
    - { name: holiday,     future_availability: known_future }

# ── Random walk (5): near-efficient financial series → naive wins ────────────
# Fixed end: "2024-06-01" ensures reproducible snapshots.

- dataset_id: sp500_weekly
  source: yfinance_multi
  source_id: "^GSPC"
  exog_tickers: ["^VIX", "^TNX", "CL=F", "GC=F", "DX-Y.NYB", "EURUSD=X", "^IXIC"]
  col_names:
    "^GSPC": sp500
    "^VIX": vix
    "^TNX": treasury_10y
    "CL=F": oil_wti
    "GC=F": gold
    "DX-Y.NYB": usd_index
    "EURUSD=X": eurusd
    "^IXIC": nasdaq
  problem_type: forecasting
  target_column: sp500
  datetime_column: week
  series_id_columns: []
  frequency: W
  forecast_horizon: 13
  interval: 1wk
  start: "2005-01-01"
  end: "2024-06-01"
  expected_family: random_walk
  exogenous_columns:
    - { name: vix,          future_availability: unknown_future }
    - { name: treasury_10y, future_availability: unknown_future }
    - { name: oil_wti,      future_availability: unknown_future }
    - { name: gold,         future_availability: unknown_future }
    - { name: usd_index,    future_availability: unknown_future }
    - { name: eurusd,       future_availability: unknown_future }
    - { name: nasdaq,       future_availability: unknown_future }

- dataset_id: oil_weekly
  source: yfinance_multi
  source_id: "CL=F"
  exog_tickers: ["NG=F", "RB=F", "HG=F", "^GSPC", "DX-Y.NYB", "GC=F", "^VIX"]
  col_names:
    "CL=F": oil_wti
    "NG=F": nat_gas
    "RB=F": gasoline
    "HG=F": copper
    "^GSPC": sp500
    "DX-Y.NYB": usd_index
    "GC=F": gold
    "^VIX": vix
  problem_type: forecasting
  target_column: oil_wti
  datetime_column: week
  series_id_columns: []
  frequency: W
  forecast_horizon: 13
  interval: 1wk
  start: "2005-01-01"
  end: "2024-06-01"
  expected_family: random_walk
  exogenous_columns:
    - { name: nat_gas,   future_availability: unknown_future }
    - { name: gasoline,  future_availability: unknown_future }
    - { name: copper,    future_availability: unknown_future }
    - { name: sp500,     future_availability: unknown_future }
    - { name: usd_index, future_availability: unknown_future }
    - { name: gold,      future_availability: unknown_future }
    - { name: vix,       future_availability: unknown_future }

- dataset_id: gold_macro_weekly
  source: yfinance_multi
  source_id: "GC=F"
  exog_tickers: ["SI=F", "HG=F", "CL=F", "DX-Y.NYB", "^TNX", "^GSPC", "^VIX"]
  col_names:
    "GC=F": gold
    "SI=F": silver
    "HG=F": copper
    "CL=F": oil_wti
    "DX-Y.NYB": usd_index
    "^TNX": treasury_10y
    "^GSPC": sp500
    "^VIX": vix
  problem_type: forecasting
  target_column: gold
  datetime_column: week
  series_id_columns: []
  frequency: W
  forecast_horizon: 13
  interval: 1wk
  start: "2005-01-01"
  end: "2024-06-01"
  expected_family: random_walk
  exogenous_columns:
    - { name: silver,       future_availability: unknown_future }
    - { name: copper,       future_availability: unknown_future }
    - { name: oil_wti,      future_availability: unknown_future }
    - { name: usd_index,    future_availability: unknown_future }
    - { name: treasury_10y, future_availability: unknown_future }
    - { name: sp500,        future_availability: unknown_future }
    - { name: vix,          future_availability: unknown_future }

- dataset_id: crypto_weekly
  source: local
  source_id: data/benchmarks/crypto_weekly.csv
  problem_type: forecasting
  target_column: btc_close
  datetime_column: week
  series_id_columns: []
  frequency: W
  forecast_horizon: 13
  expected_family: random_walk

- dataset_id: fx_weekly
  source: local
  source_id: data/benchmarks/fx_exog_weekly.csv
  problem_type: forecasting
  target_column: eurusd_close
  datetime_column: week
  series_id_columns: []
  frequency: W
  forecast_horizon: 13
  expected_family: random_walk
  exogenous_columns:
    - { name: gbpusd_close, future_availability: unknown_future }
    - { name: jpyusd_close, future_availability: unknown_future }
    - { name: gold_close,   future_availability: unknown_future }
```

- [ ] **Step 2: Verify the manifest parses and local source_ids exist**

```bash
uv run python -c "
import yaml
from pathlib import Path
m = yaml.safe_load(Path('scripts/benchmark_manifest.yaml').read_text())
fc = [e for e in m if e['problem_type'] == 'forecasting']
print(f'Forecasting entries: {len(fc)}')
for e in fc:
    if e['source'] == 'local':
        p = Path(e['source_id'])
        ok = '✓' if p.exists() else '✗ MISSING'
        print(f'  {ok} {e[\"dataset_id\"]} (local)')
    else:
        print(f'  - {e[\"dataset_id\"]} ({e[\"source\"]})')
"
```
Expected: 14 forecasting entries; crypto_weekly and fx_weekly show `✓`.

- [ ] **Step 3: Commit**

```bash
git add scripts/benchmark_manifest.yaml
git commit -m "feat(benchmark): replace 7 stale forecasting entries with 14 curated entries"
```

---

### Task 3: pool.reset_forecasting_experiences() + run_benchmark.py family check + --reset-forecasting

Adds the pool cleanup method and extends the runner with a per-record family sanity-check and the `--reset-forecasting` / `--strict` flags.

**Files:**
- Modify: `src/mlops_agents/experience/pool.py`
- Modify: `scripts/run_benchmark.py`
- Modify: `tests/test_scripts/test_benchmark_datasets.py`

- [ ] **Step 1: Append unit tests to test_benchmark_datasets.py**

```python
# ---------------------------------------------------------------------------
# Family-check unit tests
# ---------------------------------------------------------------------------

def test_family_check_statistical() -> None:
    from scripts.run_benchmark import _check_family
    assert _check_family("ets", "statistical") is True
    assert _check_family("auto_arima", "statistical") is True
    assert _check_family("naive", "statistical") is True
    assert _check_family("lightgbm_forecaster", "statistical") is False


def test_family_check_random_walk() -> None:
    from scripts.run_benchmark import _check_family
    assert _check_family("naive", "random_walk") is True
    assert _check_family("ets", "random_walk") is False      # only naive qualifies
    assert _check_family("seasonal_naive", "random_walk") is False


def test_family_check_supervised() -> None:
    from scripts.run_benchmark import _check_family
    assert _check_family("lightgbm_forecaster", "supervised") is True
    assert _check_family("random_forest_forecaster", "supervised") is True
    assert _check_family("ets", "supervised") is False
    assert _check_family("naive", "supervised") is False


def test_reset_forecasting_experiences(tmp_path) -> None:
    from mlops_agents.experience.pool import ExperiencePool
    pool = ExperiencePool(tmp_path / "test.db", audit_dir=tmp_path)
    with pool._conn() as conn:
        for task_id, problem_type in [("fc1", "forecasting"), ("cls1", "classification")]:
            conn.execute(
                """INSERT INTO experiences
                   (task_id, problem_type, dataset_profile_json,
                    training_plan_json, created_at)
                   VALUES (?, ?, '{}', '{}', datetime('now'))""",
                (task_id, problem_type),
            )
    assert pool.count("forecasting") == 1
    assert pool.count("classification") == 1

    n = pool.reset_forecasting_experiences()

    assert n == 1
    assert pool.count("forecasting") == 0
    assert pool.count("classification") == 1  # untouched
```

- [ ] **Step 2: Run failing tests to confirm they fail**

```bash
uv run pytest tests/test_scripts/test_benchmark_datasets.py::test_family_check_statistical tests/test_scripts/test_benchmark_datasets.py::test_reset_forecasting_experiences -v
```
Expected: `ImportError` for `_check_family`; `AttributeError` for `reset_forecasting_experiences`.

- [ ] **Step 3: Add reset_forecasting_experiences() to pool.py**

In `src/mlops_agents/experience/pool.py`, add this method immediately after `insert_from_record`:

```python
def reset_forecasting_experiences(self) -> int:
    """Delete all forecasting experiences and their cascade-linked rows.

    candidate_results and model_artifacts have ON DELETE CASCADE, so a single
    DELETE on experiences suffices (foreign_keys = ON is set in _conn).
    Returns the number of experience rows deleted.
    """
    with self._conn() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM experiences WHERE problem_type = 'forecasting'"
        ).fetchone()[0]
        if n == 0:
            return 0
        conn.execute("DELETE FROM experiences WHERE problem_type = 'forecasting'")
    logger.info(f"[pool] reset_forecasting: deleted {n} experiences")
    return n
```

- [ ] **Step 4: Add _check_family() and family logic to run_benchmark.py**

Add after the imports in `scripts/run_benchmark.py`:

```python
_STATISTICAL_MODELS = frozenset({"naive", "seasonal_naive", "ets", "auto_arima"})
_SUPERVISED_MODELS = frozenset({
    "random_forest_forecaster", "extra_trees_forecaster", "gbm_forecaster",
    "lightgbm_forecaster", "xgboost_forecaster", "svr_forecaster",
})


def _check_family(model_key: str, expected_family: str) -> bool:
    """True if model_key belongs to expected_family."""
    if expected_family == "statistical":
        return model_key in _STATISTICAL_MODELS
    if expected_family == "random_walk":
        return model_key == "naive"
    if expected_family == "supervised":
        return model_key in _SUPERVISED_MODELS
    return False
```

Modify `run_benchmark` signature to add two new parameters after `n_trials_override`:

```python
def run_benchmark(
    manifest_path: Path = Path("scripts/benchmark_manifest.yaml"),
    db_path: Path | None = None,
    audit_dir: Path | None = None,
    splits_dir: Path | None = None,
    staged_dir: Path | None = None,
    n_trials_override: int | None = None,
    reset_forecasting: bool = False,
    strict: bool = False,
) -> tuple[int, int]:
```

Right after `pool = ExperiencePool(db_path, audit_dir=audit_dir)`, add:

```python
    if reset_forecasting:
        n_del = pool.reset_forecasting_experiences()
        logger.info(f"[benchmark] reset_forecasting: removed {n_del} stale forecasting experiences")

    family_mismatches: list[dict] = []
```

After `pool.insert_from_record(record)` inside the loop, add:

```python
            champion_key = result.champion_candidate["model_key"]
            expected_family = entry.get("expected_family")
            if expected_family and entry.get("problem_type") == "forecasting":
                match = _check_family(champion_key, expected_family)
                symbol = "✓" if match else "✗"
                logger.info(
                    f"[{dataset_id}] family_check {symbol} "
                    f"champion={champion_key} expected_family={expected_family}"
                )
                if not match:
                    family_mismatches.append({
                        "dataset_id": dataset_id,
                        "champion": champion_key,
                        "expected_family": expected_family,
                    })
```

Add this block just before `logger.info(f"Benchmark complete...")` at the end of the function:

```python
    if family_mismatches:
        logger.warning(f"[benchmark] {len(family_mismatches)} family mismatch(es):")
        for m in family_mismatches:
            logger.warning(
                f"  ✗ {m['dataset_id']}: champion={m['champion']} "
                f"expected={m['expected_family']}"
            )
        if strict:
            n_fail += len(family_mismatches)
    else:
        logger.info("[benchmark] All forecasting family checks passed ✓")
```

Modify `main()` to wire the new flags:

```python
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("scripts/benchmark_manifest.yaml"))
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--reset-forecasting", action="store_true",
                        help="Delete all forecasting pool experiences before re-seeding")
    parser.add_argument("--strict", action="store_true",
                        help="Count family mismatches as failures (non-zero exit)")
    args = parser.parse_args()
    n_ok, n_fail = run_benchmark(
        manifest_path=args.manifest,
        n_trials_override=args.trials,
        reset_forecasting=args.reset_forecasting,
        strict=args.strict,
    )
    sys.exit(0 if n_fail == 0 else 1)
```

- [ ] **Step 5: Run all tests — should now pass**

```bash
uv run pytest tests/test_scripts/test_benchmark_datasets.py -v
```
Expected: all tests pass.

- [ ] **Step 6: Run full suite to confirm no regressions**

```bash
uv run pytest -m "not integration" -q
```
Expected: same pass count as before + new tests green.

- [ ] **Step 7: Commit**

```bash
git add src/mlops_agents/experience/pool.py scripts/run_benchmark.py \
        tests/test_scripts/test_benchmark_datasets.py
git commit -m "feat(benchmark): family sanity-check + --reset-forecasting + pool reset method"
```

---

## Running the benchmark (after all tasks complete)

```bash
uv run python scripts/run_benchmark.py --reset-forecasting --trials 10
```

Add `--strict` to make family mismatches count as failures:

```bash
uv run python scripts/run_benchmark.py --reset-forecasting --trials 10 --strict
```

The runner logs `✓`/`✗` per forecasting entry and a final summary. Any mismatch is worth investigating (wrong horizon, dataset prep issue) but some regime uncertainty is expected (e.g. `nile` is statistical but naive-adjacent).
