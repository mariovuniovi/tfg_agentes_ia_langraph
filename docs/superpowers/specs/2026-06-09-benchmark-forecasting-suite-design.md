# Curated Forecasting Benchmark Suite — Experience-Pool Enrichment

**Date:** 2026-06-09
**Status:** Design approved, pending spec review
**Scope:** Single sub-project. Independent of SP2a (model eligibility). Builds on SP1 (honest test eval) + SP2b′ (deterministic validation/exog), which the benchmark already routes through.

## Problem

The planner retrieves "similar past experiences" from the experience pool (`storage/mlops_metadata.db`) to inform model selection. The current forecasting experiences are weak priors:
- Only 7 forecasting datasets, **stale** (May-11, recorded before SP1/SP2b′ — they hold the old validation-as-champion, single-split metrics).
- 5 are **live `yfinance` scrapes** (`source: yfinance_multi`) — non-reproducible (the series grows daily; the recorded profile/champion drifts run-to-run) and flaky (API failures).
- Coverage is narrow and **statistical-skewed** — nothing teaches the planner when a *supervised* lag-based forecaster (lightgbm/xgboost/random_forest/extra_trees/gbm) is the right choice.

Result: the planner's retrieval can't distinguish the regimes where ETS/ARIMA win, where tree ensembles win, and where a flat `naive`/random-walk is genuinely best.

## Goal

Seed the experience pool with a **curated, reproducible, regime-balanced suite of famous forecasting datasets**, run through the (now honest) benchmark so the recorded experiences carry trustworthy profiles + champions + scores. Cover three distinct regimes so retrieval can separate them.

## Key decisions (from brainstorming)

1. **Honest-run + sanity-check** (not hand-authored). Each dataset is run through the real benchmark; the champion is whatever honestly wins the full eligible pool. A per-dataset `expected_family` is asserted as a *validation gate* (does the pipeline reproduce the literature-known regime?), not injected.
2. **Bundled local CSVs** for reproducibility. Every dataset is fetched once from its canonical source, trimmed, resampled, and committed under `data/benchmarks/` as `source: local`. No network at benchmark runtime. Financial series are **frozen snapshots** (fetched once at a fixed end-date), not live downloads.
3. **Three regimes, ~14 datasets.** Statistical-smooth, supervised-exog, financial near-random-walk.
4. **Reuse the existing benchmark engine** (`run_benchmark.py` + `benchmark_manifest.yaml`); no separate manifest.
5. **Refresh:** clear the stale forecasting experiences from the pool DB and repopulate from the curated suite. Classification/regression experiences untouched.
6. **Datasets chosen to fit our model set + policies (verified against the registry):** every expected champion is a registry model; supervised-regime datasets are sized ≥500 rows so `lightgbm`/`xgboost` clear the `min_rows` eligibility bucket — supervised models win from lagged target values + ETS-extended weather/sensor exog; statistical-regime datasets are short and univariate so statistical models win; financial datasets are near-random-walk so `naive` wins. The deterministic validation + exog policies apply uniformly to every dataset.

## The suite (14 datasets, 3 regimes)

| dataset_id | freq | regime / `expected_family` | exog | expected champion | bundle source |
|---|---|---|---|---|---|
| air_passengers | MS | statistical | no | ets / auto_arima | already in `data/benchmarks/` |
| co2_mauna_loa | MS | statistical | no | ets / auto_arima | `statsmodels.datasets.co2` (resample W→MS) |
| sunspots | YS | statistical | no | auto_arima | `statsmodels.datasets.sunspots` |
| nile | YS | statistical (random-walk-ish) | no | auto_arima / naive | `statsmodels.datasets.nile` |
| bike_sharing_daily | D | supervised | yes (temp, hum, windspeed, season, holiday, workingday, weekday) | tree ensemble | OpenML 42712 (daily `cnt`) |
| metro_traffic_volume | D | supervised | yes (temp, rain, snow, clouds, holiday) | lightgbm / xgboost | UCI 492 (resample H→D) |
| beijing_pm25 | D | supervised | yes (dew point, temp, pressure, wind, precip) | random_forest / gbm | UCI 381 (resample H→D) |
| appliances_energy | H | supervised | yes (room temp/humidity, outdoor weather) | tree ensemble | UCI 374 (resample 10min→H) |
| vic_elec | D | supervised | yes (temperature, holiday) | lightgbm / xgboost | `vic_elec` public CSV mirror (resample 30min→D) |
| sp500_weekly | W | random_walk | macro exog (frozen) | naive / random-walk | yfinance @ fixed end-date |
| oil_weekly | W | random_walk | macro exog (frozen) | naive / random-walk | yfinance @ fixed end-date |
| gold_macro_weekly | W | random_walk | macro exog (frozen) | naive / random-walk | yfinance @ fixed end-date |
| crypto_weekly | W | random_walk | no | naive / random-walk | yfinance @ fixed end-date |
| fx_weekly | W | random_walk | macro exog (frozen) | naive / random-walk | yfinance @ fixed end-date |

`expected_family` values: **statistical** = {naive, seasonal_naive, ets, auto_arima}; **supervised** = {lightgbm_forecaster, xgboost_forecaster, gbm_forecaster, random_forest_forecaster, extra_trees_forecaster, svr_forecaster}; **random_walk** = {naive}. (random_walk is a *subset* of statistical — it asserts the *flat* baseline specifically wins, which is the meaningful financial-regime signal.)

**Per-dataset frequency + forecast_horizon are finalized at bundling time** to satisfy the SP2b′ capacity floor (`n_obs ≥ min_train + 2*horizon`, `min_train = max(3*horizon, 30)`); the hourly/sub-daily sources are resampled (to D or H) both to bound series length and to keep V1 single-series. Resampling rule per dataset is fixed in the fetch script (deterministic).

## Components

### 1. `scripts/fetch_benchmark_datasets.py` (new, run once)

One function per dataset: fetch from its canonical source, trim to `[datetime, target, exog…]`, resample to the target frequency, write `data/benchmarks/<dataset_id>.csv`. Deterministic: fixed yfinance end-date constant for the 5 financial snapshots; fixed resampling aggregations (e.g. mean for weather, sum/mean for demand). Idempotent (rewrites the same CSV). A `--only <dataset_id>` flag to refetch one. Tolerates a source being unavailable by skipping with a clear message (so one flaky source doesn't block the rest), reporting which CSVs were (re)written.

No derived calendar columns (month, dayofweek, hour) — the date column is the only datetime artifact needed. Supervised models win from lagged target values + ETS-extended weather/sensor exog, which provides sufficient signal over univariate ETS/ARIMA on the raw target. Sizing rule: supervised datasets must land in the `small`+ row bucket (≥500 rows) so `lightgbm`/`xgboost` are registry-eligible — hence `appliances_energy` is resampled to **hourly** (~3,300 rows), not daily (~137).

### 2. `benchmark_manifest.yaml` (modify)

Replace the current 7-entry forecasting section with the 14 curated entries, all `source: local` pointing at `data/benchmarks/<id>.csv`, each with `problem_type: forecasting`, `target_column`, `datetime_column`, `frequency`, `forecast_horizon`, `exogenous_columns` (list of `{name, future_availability}` where applicable; weather/sensor columns are `unknown_future`; structured categorical columns like `holiday`/`season` that ship with the dataset and are known in advance are `known_future`), and `expected_family`. Classification/regression entries unchanged.

### 3. `run_benchmark.py` (modify — family sanity-check)

After ingesting each forecasting record, map the recorded champion `model_key` to its family and compare to the entry's `expected_family`. Log `✓ <dataset_id>: champion=<model> (family matches)` or `✗ <dataset_id>: champion=<model> family=<actual> expected=<expected>`. Print a summary table at the end (N matched / M mismatched). Add a `--strict` flag that exits non-zero if any forecasting mismatch occurs. The family map is a small dict in the runner (or a shared helper). Non-forecasting entries skip the check.

### 4. Pool refresh

Before/at repopulation, delete the existing **forecasting** experiences from the pool DB so stale May-11 records (and the old live-yfinance financial ones) don't linger or duplicate. A small one-off step (a `--reset-forecasting` flag on the runner, or a documented `DELETE FROM experiences WHERE problem_type='forecasting'` + cascade to `candidate_results`/`model_artifacts`). Classification/regression experiences are preserved. Then run the curated suite to repopulate.

## Data flow

```
fetch_benchmark_datasets.py (once) → data/benchmarks/*.csv (committed)
run_benchmark.py:
  reset forecasting experiences in pool DB
  for each manifest entry:
    fetch_dataset (source: local → read CSV) → build_dataset_profile
    → default_training_plan (full eligible pool)
    → run_training_plan (honest test eval + deterministic validation/exog; forecasting_settings=None → SP2b′ fallback)
    → ExperienceRecord → pool.insert_from_record  (writes experiences + candidate_results + model_artifacts + JSON audit)
    → family sanity-check vs expected_family
  print pass/mismatch summary
```

## Testing / verification

- **Bundled-CSV smoke test:** a light test (or a `--check` mode of the fetch script) asserting each committed `data/benchmarks/<id>.csv` exists, parses, has the manifest's `datetime_column` + `target_column` + declared exog columns, and a sane row count (≥ the capacity floor for its horizon). Runs offline against committed CSVs (no network).
- **Family sanity-check:** the run-time gate in §3 is the primary correctness signal; the human-reviewed summary + optional `--strict` is the gate. Mismatches are investigated (dataset prep, horizon, or model-pool issue), not silently accepted.
- **No new unit tests for training** — the benchmark run itself is the validation; it must not be in the CI unit suite (slow, network-for-bundling). Per testing conventions, any added pytest is offline + uses the committed CSVs.

## Non-goals
- Changing the planner, retrieval, or model registry (separate from SP2a).
- Multi-series/panel datasets (V1 is single-target).
- Live/auto-refreshing datasets (frozen snapshots only).
- Hand-authored experience records.

## Success criteria
- 14 committed `data/benchmarks/*.csv` forecasting datasets across the 3 regimes; benchmark runs offline (no network) and is reproducible run-to-run.
- The pool's forecasting experiences are exactly these 14, with honest (SP1/SP2b′) profiles + champions + test scores.
- The family sanity-check passes for the clear-cut datasets (statistical ones → statistical champion; financial ones → naive/random-walk); each supervised-regime dataset's champion is in the supervised family (or any mismatch is documented with a reason).
- At least one dataset per family produces that family's champion, so the planner's retrieval has genuine ETS/ARIMA, tree-ensemble, AND random-walk exemplars.
