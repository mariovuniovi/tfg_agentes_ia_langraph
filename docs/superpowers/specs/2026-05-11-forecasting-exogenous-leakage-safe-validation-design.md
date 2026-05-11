# Forecasting: Exogenous-Variable Handling & Leakage-Safe Validation — Design Spec

**Date:** 2026-05-11
**Status:** Approved (pending final user sign-off)
**Sub-project:** SP4 extension (post-experience-pool seeding)

---

## 1. Goal

Add explicit support for exogenous variables with declared future-availability and a leakage-safe validation loop. The executor must never use realized future exogenous values during validation unless those values would genuinely be available at prediction time. Per-column extension strategies (`naive_carry`, `ets`, `auto_arima`, `drop`) are proposed by the user/manifest/LLM and enforced deterministically by the executor.

## 2. Motivation

The current executor (post the multi-exog datasets sprint) silently feeds **realized future exogenous values** into both training and prediction. This is temporal leakage: the validation metric reflects "what would happen if we knew oil prices in advance" rather than production conditions. For commodity/financial forecasting — the project's core use case — this distorts model selection and over-promises accuracy.

The fix requires three changes that must land together to avoid regressions:
1. A typed contract for declaring whether each exogenous column is **known** at prediction time or **must be forecast first**.
2. A leakage-safe validation loop that extends unknown-future columns from training-window history before feeding them to the target model.
3. A typed validation strategy (`single_split` / `rolling_window` / `expanding_window`) chosen by deterministic policy from the dataset profile.

## 3. Architecture (two-layer separation)

Three contracts, three responsibilities:

| Layer | Owns | Captured when | Filled by |
|---|---|---|---|
| **Dataset schema** (`task_metadata`) | `future_availability` per exog column; `expected_drift` business hint | Data-validation time | User via UI (only when forecasting + exog present), benchmark manifest, or LLM data-agent |
| **Training plan** (`TrainingPlan.forecasting_settings`) | Extension strategy per unknown-future column; validation strategy | Plan-build time | Default policy / advanced UI / LLM planner (SP5) |
| **Executor** | Leakage-safe enforcement; per-fold loop; metric aggregation | Run-time | — |

The LLM never bypasses leakage protection. It only declares intent; the executor reads the plan, validates it, runs the right loop.

## 4. Data contracts

### 4.1 `task_metadata` (forecasting only)

Adds two optional fields:

```python
"exogenous_columns": [                                      # optional
    {"name": "holiday_flag",  "future_availability": "known_future"},
    {"name": "oil_price",     "future_availability": "unknown_future"},
    {"name": "temperature",   "future_availability": "unknown_future"},
],
"expected_drift": "low" | "medium" | "high",                # optional, defaults to "low"
```

**Resolution rules:**
- `exogenous_columns` omitted → all non-target/non-date/non-series-id columns are treated as `unknown_future` exog (leakage-safe default).
- `exogenous_columns` present → **authoritative**: only listed columns are exog; unlisted non-target columns are dropped before any modelling.
- The data validator may auto-tag calendar-derived columns (`year`, `month`, `dayofweek`, `is_holiday`) as `known_future` without prompting the user.
- `expected_drift` defaults to `"low"` if absent. Source-of-truth lives in `task_metadata`, not in `DatasetProfile`.

### 4.2 `TrainingPlan.forecasting_settings`

Replaces the currently empty forecasting settings with three typed Pydantic models:

```python
class ValidationStrategy(BaseModel):
    type: Literal["single_split", "rolling_window", "expanding_window"] = "single_split"
    n_folds: int = 1                  # ignored for single_split
    horizon: int                       # must equal task_metadata.forecast_horizon
    step_size: int | None = None       # defaults to horizon
    window_size: int | None = None     # only used by rolling_window; None → auto

ExogStrategy = Literal["known_future", "naive_carry", "ets", "auto_arima", "drop"]

class ExogStrategySettings(BaseModel):
    per_column: dict[str, ExogStrategy] = {}
    default_unknown_future: Literal["naive_carry", "ets", "auto_arima", "drop"] = "naive_carry"

class ForecastingSettings(BaseModel):
    validation_strategy: ValidationStrategy
    exog_strategies: ExogStrategySettings
```

**Resolution rules:**
- A column not present in `per_column` and tagged `unknown_future` in the schema uses `default_unknown_future`.
- A column not present in `per_column` and tagged `known_future` in the schema is always `known_future` regardless of plan.
- `per_column` keys must be valid exogenous columns from the schema. Otherwise → `ValueError`.
- A `known_future` schema column cannot be overridden by `per_column` to anything else. Business truth wins over modelling choice. Otherwise → `ValueError`.

### 4.3 `DatasetProfile` (Pydantic-ified)

Currently `build_dataset_profile()` returns a dict. Convert to a Pydantic model with `model_dump()` / `model_dump_json()` for SQLite serialization.

Adds one forecasting-only field:
```python
history_length: Literal["very_short", "short", "medium", "long"] | None
```

For single-target series, derived from total row count. For multi-target panel data, derived from the **minimum per-series length** (the bottleneck). Bucketing thresholds match existing `n_rows` conventions but interpreted in terms of time steps.

Existing `n_rows` field stays for tabular problems.

### 4.4 `ExperienceRecord` additions

Five new optional fields (clean Python names, JSON-serialized to SQLite):

```python
validation_strategy: dict | None         # what was used
exog_availability:   dict | None         # business truth at training time
exog_strategies:     dict | None         # modelling choice applied
per_fold_metrics:    list[dict] | None   # so SP5 can reason about stability
exog_fit_failures:   list[dict] | None   # logged ETS/AutoARIMA failures + fallback
```

## 5. Validation policy (deterministic)

New module: `src/mlops_agents/training/validation_policy.py`

```python
def select_validation_strategy(
    profile: DatasetProfile,
    task_metadata: dict[str, Any],
) -> ValidationStrategy:
    horizon = task_metadata["forecast_horizon"]
    history = profile.history_length
    drift   = task_metadata.get("expected_drift", "low")

    if drift == "high":
        return ValidationStrategy(
            type="rolling_window", n_folds=3, horizon=horizon,
            step_size=horizon, window_size=None,  # auto
        )
    if history in ("very_short", "short"):
        return ValidationStrategy(type="single_split", n_folds=1, horizon=horizon)
    return ValidationStrategy(
        type="expanding_window", n_folds=3, horizon=horizon, step_size=horizon,
    )


def resolve_rolling_window_size(
    total_history: int,
    horizon: int,
    n_folds: int,
    season_length: int | None,
) -> int:
    # MVP: ignore season_length. TODO: max(3*horizon, 2*season_length, 50).
    base = max(3 * horizon, 50)
    upper = total_history - n_folds * horizon
    return min(base, max(upper, horizon))


def validate_forecasting_plan(
    plan: TrainingPlan,
    task_metadata: dict[str, Any],
    profile: DatasetProfile,
    train_pool_stats: dict[str, Any],   # {"single_series": bool, "series_lengths": dict[str, int] | None, "total_len": int}
) -> None:
    """Raise ValueError on any leakage / capacity / type violation."""
```

**`validate_forecasting_plan` enforces:**

1. `validation_strategy.horizon == task_metadata["forecast_horizon"]`.
2. `validation_strategy.n_folds ≥ 1`; equals 1 iff `type == "single_split"`.
3. Capacity check (single-target):
   `total_len ≥ n_folds * horizon + max(3 * horizon, 30)`
4. Capacity check (multi-target panel):
   - Compute `too_short = {sid: L for sid, L in series_lengths.items() if L < required_len}`.
   - If `len(too_short) / n_series > 0.5` → raise `ValueError`.
   - Else log warning naming the dropped series; data preparation step removes them from `train_pool` before any modelling.
5. Every key in `plan.forecasting_settings.exog_strategies.per_column` exists in `task_metadata["exogenous_columns"]` AND has `future_availability == "unknown_future"`.
6. `rolling_window` plans whose `window_size is None` get auto-resolved via `resolve_rolling_window_size`. Explicit `window_size` must satisfy `horizon ≤ window_size ≤ total_len − n_folds*horizon`.

**MVP `min_train_len`:** `max(3 * horizon, 30)`. (TODO comment: `max(3*horizon, 2*season_length, 30)` once season_length lives in the profile.)

## 6. Executor flow (the leakage-safe loop)

Two new modules, plus modifications to `executor.py`:

```
src/mlops_agents/training/
├── validation_policy.py    (new)
├── validation_folds.py     (new)
└── exog_extender.py        (new)
```

### 6.1 `validation_folds.py`

```python
def iter_folds(
    train_pool: pd.DataFrame,
    strategy: ValidationStrategy,
    dt_col: str,
    sid_cols: list[str],
) -> Iterator[tuple[pd.Index, pd.Index]]:
    """Yield (train_idx, val_idx) pairs in chronological order.

    single_split   → 1 pair: train=[:-horizon], val=[-horizon:]
    expanding      → K pairs: train_end shifts back by step_size each fold,
                     train_start stays at 0
    rolling        → K pairs: same step_size, but train_start moves so
                     train length stays = window_size
    """
```

### 6.2 `exog_extender.py`

```python
def extend_exog(
    history: pd.Series,
    horizon: int,
    strategy: Literal["naive_carry", "ets", "auto_arima"],
    freq: str | None,
) -> tuple[pd.Series, dict | None]:
    """Return (predicted_future_values, failure_info_or_None).

    naive_carry:  repeat history.iloc[-1] for `horizon` steps; never fails.
    ets:          AutoETS().fit(history).forecast(horizon); on failure → fall
                  back to naive_carry, return failure_info dict.
    auto_arima:   AutoARIMA().fit(history).forecast(horizon); same fallback.
    """


def _align_val_exog_index(
    val_exog: pd.DataFrame,
    series_dict: dict[str, pd.Series],
    train_len: int,
    dt_col: str,
    freq: str | None,
) -> pd.DataFrame:
    """Match val_exog's index to the series_dict's index type.

    skforecast requires train_exog and val_exog to share the same index
    type as series. Picks DatetimeIndex (continuing from the last training
    date at `freq` cadence) or RangeIndex (starting at `train_len`) by
    inspecting a sample series from series_dict.
    """
```

This is the **leakage firewall**. It only ever receives `cand_train[col]` history; the executor cannot construct `val_exog` for unknown-future columns through any other path.

### 6.3 Modified `_run_candidate_forecasting` (single-target only)

```
plan_level_guard(plan, task_metadata, profile, train_pool_stats)  # may raise

strategy = plan.forecasting_settings.validation_strategy

for fold_id, (train_idx, val_idx) in enumerate(iter_folds(train_pool, strategy, ...)):
    cand_train = train_pool.loc[train_idx]
    cand_val   = train_pool.loc[val_idx]

    future_values: dict[str, pd.Series] = {}
    failures: list[dict] = []

    for col in exog_columns:
        avail = availability_map[col]
        if avail == "known_future":
            future_values[col] = cand_val[col].reset_index(drop=True)
            continue

        strat = exog_strategies.per_column.get(col, exog_strategies.default_unknown_future)
        if strat == "drop":
            continue  # column omitted from both train and val exog

        cache_key = (col, strat, fold_id, "default")
        if cache_key in exog_cache:
            future_values[col] = exog_cache[cache_key]
        else:
            preds, fail_info = extend_exog(cand_train[col], horizon, strat, freq)
            future_values[col] = preds
            exog_cache[cache_key] = preds
            if fail_info:
                failures.append(fail_info | {"fold_id": fold_id, "column": col})

    train_exog = cand_train[list(future_values.keys())]                 # realized
    val_exog   = _align_val_exog_index(                                  # predicted / known
        pd.DataFrame(future_values),
        series_dict,           # determines index type used by the forecaster
        train_len=len(cand_train),
        dt_col=dt_col,
        freq=freq,
    )

    # Invariant: train_exog.columns == val_exog.columns (preserved by `drop` removing from both)
    assert list(train_exog.columns) == list(val_exog.columns)

    forecaster.fit(series=series_dict_from(cand_train), exog=train_exog)
    preds = forecaster.predict(steps=horizon, exog=val_exog)
    fold_score = score(cand_val[target], preds)

trial_score = mean(fold_scores)
return trial_score
```

**Caching:** `exog_cache` is scoped to one `_run_candidate_forecasting` call (one model candidate). It is **NOT shared across candidates** because different candidates may have different `exog_strategies` via plan overrides (rare but possible). Within a candidate, all Optuna trials share the cache (exog forecasts depend on `cand_train` only, not on the target model's hyperparameters).

**For 7 exog × 3 folds × 8 trials**, the cache reduces 168 exog fits to 21 per candidate.

### 6.4 Multi-target (panel) datasets

Out of scope for v1. When `sid_cols` is non-empty:
- `validate_forecasting_plan` accepts the plan as long as `per_column` is empty and `default_unknown_future == "naive_carry"`. Anything else raises `NotImplementedError("Multi-target exog support deferred to v2").`
- `_run_candidate_forecasting` continues to call `_build_exog_df(...)` which returns `None` for `sid_cols` (current behavior). No regression to existing panel benchmarks.

### 6.5 Failure modes

| Failure | Action |
|---|---|
| ETS/AutoARIMA fit fails on one exog column | Fall back to `naive_carry` for that column+fold; append to `exog_fit_failures` with `{column, strategy, fold_id, error_class, error_msg}` |
| Main forecaster fit fails for a fold | Trial returns `inf` (existing Optuna failure path) |
| Plan validation fails | `validate_forecasting_plan` raises `ValueError` before Optuna study creation |

## 7. ML rules YAML extensions

Add to `knowledge/ml_rules.yaml` a new `forecasting_rules` section. These are **planner guidance for SP5** — not executable defaults. The deterministic Python policy in §5 is the source of truth at runtime.

```yaml
forecasting_rules:
  # ─── Validation strategy ──────────────────────────────────────────
  - rule_id: forecasting_short_history_single_split
    applies_when: { problem_type: forecasting, history_length: [very_short, short] }
    recommend:    { validation_strategy: single_split }
    reason: "Multiple folds leave too little data per fold with short history."

  - rule_id: forecasting_medium_long_expanding_window
    applies_when: { problem_type: forecasting, history_length: [medium, long] }
    recommend:    { validation_strategy: expanding_window }
    reason: "Sufficient history makes expanding-window backtesting more robust."

  - rule_id: forecasting_high_drift_rolling_window
    applies_when: { problem_type: forecasting, expected_drift: high }
    recommend:    { validation_strategy: rolling_window }
    reason: "Non-stationary processes benefit from a fixed-size recent training window."

  # ─── Exogenous strategy ────────────────────────────────────────────
  - rule_id: exog_calendar_known_future
    applies_when: { problem_type: forecasting, exog_column_kind: calendar_derived }
    recommend:    { future_availability: known_future }
    reason: "Calendar features (year, month, dayofweek, is_holiday) are deterministic and known ahead."

  - rule_id: exog_unknown_default_naive_carry
    applies_when: { problem_type: forecasting, exog_future_availability: unknown_future }
    recommend:    { exog_strategy: naive_carry }
    reason: "Safest default; cheap; competitive for short horizons."

  - rule_id: exog_slow_macro_auto_arima
    applies_when: { problem_type: forecasting, exog_kind: macro_indicator, history_length: [medium, long] }
    recommend:    { exog_strategy: auto_arima }
    reason: "Slow-moving macro variables (rates, FX) have ARIMA-friendly dynamics; outperform naive over longer horizons."
```

The rule matcher receives a merged context: `{**profile.model_dump(), "expected_drift": task_metadata.get("expected_drift", "low")}`. Rules can match on profile fields plus selected task_metadata fields.

`exog_column_kind` and `exog_kind` are optional hints attached to columns by the LLM data-agent in SP5; not required in v1 — the deterministic defaults work without them.

## 8. Experience pool migration

`storage/mlops_metadata.db` migration adds five nullable TEXT columns to `experiences`:

```sql
ALTER TABLE experiences ADD COLUMN validation_strategy_json TEXT;
ALTER TABLE experiences ADD COLUMN exog_availability_json   TEXT;
ALTER TABLE experiences ADD COLUMN exog_strategies_json     TEXT;
ALTER TABLE experiences ADD COLUMN per_fold_metrics_json    TEXT;
ALTER TABLE experiences ADD COLUMN exog_fit_failures_json   TEXT;
```

Existing rows (pre-migration) remain valid with NULL in these columns. The pool's `insert_from_record` uses `INSERT OR REPLACE`, so re-seeding the benchmark after migration produces clean records.

A one-time data-fix step re-seeds the existing 21 benchmark records by running `scripts/run_benchmark.py` again (now with leakage-safe validation). Champions and scores will change for any dataset whose previous champion benefited from exog leakage.

## 9. MLflow integration

Parent-run params added per training run:
- `validation_strategy_type`
- `validation_n_folds`
- `exog_default_strategy`
- `expected_drift`

Per-fold scores logged as sequential metrics: `fold_0_<metric>`, `fold_1_<metric>`, ..., plus `fold_mean_<metric>` and `fold_std_<metric>`.

## 10. Testing strategy

### 10.1 Unit tests (no LLM)

| Module | Coverage |
|---|---|
| `validation_policy.py` | `select_validation_strategy` for each (history_length × expected_drift) combo; `validate_forecasting_plan` raises on each violation class; `resolve_rolling_window_size` bounds |
| `validation_folds.py` | `iter_folds` for each strategy: correct count, no overlap with future, chronological order, length invariants |
| `exog_extender.py` | naive_carry deterministic; ETS/AutoARIMA on synthetic series; failure fallback; identical history → identical output (cache correctness) |

### 10.2 Integration tests (deterministic, no LLM)

| Test | Assertion |
|---|---|
| End-to-end single-target with 3 unknown_future exog + 1 known_future + 1 drop | Champion selected; `val_exog` columns exclude `drop`; `known_future` column matches realized values; `unknown_future` columns match their extender output |
| End-to-end with `expanding_window` K=3 | Three folds run; `per_fold_metrics` has length 3; aggregate is mean |
| Multi-target panel dataset | `NotImplementedError` raised if plan has non-empty `per_column` or non-naive_carry default; otherwise runs as today with no exog |
| Plan with `per_column` key not in schema | `ValueError` before Optuna study creation |
| Plan attempting to override a `known_future` column | `ValueError` before Optuna study creation |
| Plan with insufficient history for K folds | `ValueError` naming required vs actual length |
| ETS fit failure on synthetic edge case | Trial succeeds; `exog_fit_failures` contains one entry; that column uses naive_carry for that fold |

### 10.3 Regression test

Re-run the full benchmark (21 datasets) and confirm:
- All 21 still complete with no errors.
- Datasets with declared exog now show **per_fold_metrics** populated.
- Datasets without declared exog (forecasting datasets without `exogenous_columns` in manifest) get auto-tagged all-unknown_future and use `naive_carry` defaults — scores may shift from prior leakage-allowing runs; this is expected and correct.

## 11. Acceptance criteria

1. ✅ All 21 benchmark datasets complete via `scripts/run_benchmark.py --trials 8` without errors.
2. ✅ Experience pool records contain the five new fields populated for forecasting tasks.
3. ✅ Re-running the benchmark twice produces identical records (deterministic).
4. ✅ A regression test confirms no realized future exog value reaches the main forecaster's `predict()` call for any `unknown_future` column.
5. ✅ All existing 274 tests still pass.
6. ✅ MLflow runs show the new parent-run params and per-fold metrics.
7. ✅ Documentation (`CLAUDE.md` if applicable) updated to describe `exogenous_columns` and `forecasting_settings` schema.

## 12. Explicitly out of scope (v1)

- Per-column ETS/AutoARIMA for **multi-target (panel) datasets**. Multi-target exog uses `naive_carry` only or raises `NotImplementedError`. v2.
- `scenario_based`, `market_implied`, `forecasted` covariate types from the original brief. The `known_future` / `unknown_future` binary covers ≥95% of real cases; richer types deferred.
- Static covariates (series-level metadata for hierarchical forecasting).
- Auto-detection of `expected_drift` from the time series itself. v1 = user-provided business hint.
- UI flow for prompting the user. The data-validation UI is a separate work stream; this spec assumes `task_metadata` arrives already populated.
- Optuna-tuning of `n_folds` / `window_size`. These are evaluation protocol, not model hyperparameters. The `build_suggest_fn` is built from `spec.search_space` only and cannot see validation-strategy params.
- Season-length-aware `min_train_len`. MVP uses `max(3 × horizon, 30)`; seasonal version is a TODO comment.

## 13. Files touched (summary)

| Path | Change |
|---|---|
| `src/mlops_agents/contracts/training.py` | Add `ValidationStrategy`, `ExogStrategySettings`, `ForecastingSettings`; wire into `TrainingPlan` |
| `src/mlops_agents/contracts/task_metadata.py` (or wherever schema lives) | Add `exogenous_columns`, `expected_drift` documentation |
| `src/mlops_agents/training/profiler.py` | Convert `DatasetProfile` to Pydantic model; add `history_length` |
| `src/mlops_agents/training/validation_policy.py` | **New module** |
| `src/mlops_agents/training/validation_folds.py` | **New module** |
| `src/mlops_agents/training/exog_extender.py` | **New module** |
| `src/mlops_agents/training/executor.py` | Modify `_run_candidate_forecasting`; remove leakage-allowing `_build_exog_df(val, ...)` call; add cache; call validation_policy + iter_folds + extend_exog |
| `src/mlops_agents/experience/schema.py` | Add five new optional fields to `ExperienceRecord` |
| `src/mlops_agents/experience/pool.py` | SQL migration (idempotent `ADD COLUMN IF NOT EXISTS`-style); update insert |
| `knowledge/ml_rules.yaml` (or wherever rules live) | Add 6 new rules |
| `scripts/benchmark_manifest.yaml` | Add `exogenous_columns` blocks to forecasting entries; `expected_drift` for financial datasets |
| `scripts/run_benchmark.py` | Pass `expected_drift` through to `task_metadata` |
| `tests/` | New tests per §10 |
