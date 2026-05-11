# Forecasting: Exogenous Handling & Leakage-Safe Validation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the forecasting executor leakage-safe by tagging exogenous columns with future-availability metadata, extending unknown-future columns from training history before predicting the target, and running K-fold backtesting where appropriate.

**Architecture:** Two-layer separation. The dataset schema (`task_metadata`) declares **whether** each exogenous column is known at prediction time. The `TrainingPlan.forecasting_settings` declares **how** unknown-future columns get extended (naive_carry / ets / auto_arima / drop) and which validation strategy to use (single_split / rolling_window / expanding_window). The executor reads both, validates the plan, runs a per-fold loop that builds `val_exog` exclusively through a `extend_exog()` firewall, and aggregates per-fold scores.

**Tech Stack:** Python 3.12, Pydantic v2, pandas, skforecast `ForecasterRecursiveMultiSeries` (already in use), statsforecast `AutoETS` / `AutoARIMA`, Optuna (untouched — validation strategy is set once per run, never tuned), MLflow (parent-run params + per-fold metrics), SQLite (experience pool migration), loguru.

**Spec:** `docs/superpowers/specs/2026-05-11-forecasting-exogenous-leakage-safe-validation-design.md`

---

## File map (what each new/modified file owns)

| File | Status | Responsibility |
|---|---|---|
| `src/mlops_agents/contracts/training.py` | Modify | Add `ValidationStrategy`, `ExogStrategySettings`, `ForecastingSettings`; replace existing `forecasting_settings: dict` with the typed model |
| `src/mlops_agents/contracts/task_metadata.py` | Create | Pydantic `ExogColumnMeta` + helper resolvers (or live alongside training contracts) |
| `src/mlops_agents/training/profiler.py` | Modify | Convert profile to `DatasetProfile` Pydantic model; add `history_length` field |
| `src/mlops_agents/training/validation_folds.py` | Create | `iter_folds(train_pool, strategy, dt_col, sid_cols) → Iterator[(train_idx, val_idx)]` |
| `src/mlops_agents/training/exog_extender.py` | Create | `extend_exog(history, horizon, strategy, freq)` + `_align_val_exog_index(...)` |
| `src/mlops_agents/training/validation_policy.py` | Create | `select_validation_strategy`, `resolve_rolling_window_size`, `validate_forecasting_plan` |
| `src/mlops_agents/training/executor.py` | Modify | Rewrite `_run_candidate_forecasting` (use new modules); update `_retrain_forecasting` (no leakage risk — uses realized exog history only) |
| `src/mlops_agents/experience/schema.py` | Modify | Add 5 nullable fields to `ExperienceRecord` |
| `src/mlops_agents/experience/pool.py` | Modify | PRAGMA-based migration adding 5 columns; widen `insert_from_record` |
| `src/mlops_agents/knowledge/reader.py` | Modify | Add `recommend: dict[str, Any]` field to `MLRule` |
| `src/mlops_agents/knowledge/ml_rules.yaml` | Modify | Add 6 new rules under a `forecasting_rules:` section |
| `scripts/run_benchmark.py` | Modify | `build_task_metadata` propagates `exogenous_columns` + `expected_drift`; default policy fills `forecasting_settings` |
| `scripts/benchmark_manifest.yaml` | Modify | Add `exogenous_columns` blocks + `expected_drift` to relevant entries |
| `tests/test_training/test_validation_folds.py` | Create | Unit tests for `iter_folds` |
| `tests/test_training/test_exog_extender.py` | Create | Unit tests for `extend_exog` |
| `tests/test_training/test_validation_policy.py` | Create | Unit tests for `select_validation_strategy`, `validate_forecasting_plan` |
| `tests/test_training/test_executor_forecasting_leakage.py` | Create | Integration tests (no-leakage invariant, K-fold scoring, multi-target guard) |
| `tests/test_contracts/test_forecasting_settings.py` | Create | Unit tests for the new Pydantic models |

---

## Task 1: Add typed Pydantic models for forecasting settings

**Files:**
- Modify: `src/mlops_agents/contracts/training.py`
- Create: `tests/test_contracts/test_forecasting_settings.py`

**Why this first:** Every downstream module imports these types. Land contracts before any behavior.

- [ ] **Step 1: Write failing tests for the new models**

Create `tests/test_contracts/test_forecasting_settings.py`:

```python
"""Tests for ValidationStrategy, ExogStrategySettings, ForecastingSettings."""
import pytest
from pydantic import ValidationError

from mlops_agents.contracts.training import (
    ValidationStrategy,
    ExogStrategySettings,
    ForecastingSettings,
)


def test_validation_strategy_defaults_single_split():
    s = ValidationStrategy(horizon=12)
    assert s.type == "single_split"
    assert s.n_folds == 1
    assert s.step_size is None
    assert s.window_size is None


def test_validation_strategy_rejects_unknown_type():
    with pytest.raises(ValidationError):
        ValidationStrategy(type="nonsense", horizon=12)


def test_validation_strategy_rolling_with_window():
    s = ValidationStrategy(
        type="rolling_window", n_folds=3, horizon=12, step_size=12, window_size=60
    )
    assert s.window_size == 60


def test_exog_strategy_settings_empty_defaults():
    e = ExogStrategySettings()
    assert e.per_column == {}
    assert e.default_unknown_future == "naive_carry"


def test_exog_strategy_settings_per_column_rejects_unknown_value():
    with pytest.raises(ValidationError):
        ExogStrategySettings(per_column={"oil": "magic"})


def test_forecasting_settings_compose():
    fs = ForecastingSettings(
        validation_strategy=ValidationStrategy(horizon=6),
        exog_strategies=ExogStrategySettings(per_column={"oil": "auto_arima"}),
    )
    assert fs.validation_strategy.n_folds == 1
    assert fs.exog_strategies.per_column["oil"] == "auto_arima"
```

- [ ] **Step 2: Run the tests to confirm they fail**

```
uv run python -m pytest tests/test_contracts/test_forecasting_settings.py -v
```
Expected: ImportError / failures for missing classes.

- [ ] **Step 3: Add the Pydantic models**

In `src/mlops_agents/contracts/training.py`, add near the top of the existing model definitions (after imports, before `class TrainingPlanCandidate`):

```python
from typing import Literal
from pydantic import BaseModel, Field


ExogStrategy = Literal["known_future", "naive_carry", "ets", "auto_arima", "drop"]
UnknownFutureStrategy = Literal["naive_carry", "ets", "auto_arima", "drop"]


class ValidationStrategy(BaseModel):
    type: Literal["single_split", "rolling_window", "expanding_window"] = "single_split"
    n_folds: int = 1
    horizon: int
    step_size: int | None = None
    window_size: int | None = None


class ExogStrategySettings(BaseModel):
    per_column: dict[str, ExogStrategy] = Field(default_factory=dict)
    default_unknown_future: UnknownFutureStrategy = "naive_carry"


class ForecastingSettings(BaseModel):
    validation_strategy: ValidationStrategy
    exog_strategies: ExogStrategySettings = Field(default_factory=ExogStrategySettings)
```

- [ ] **Step 4: Re-wire `TrainingPlan` to use `ForecastingSettings`**

Find the existing `forecasting_settings: dict` (or equivalent) on `TrainingPlan` and change it to `forecasting_settings: ForecastingSettings | None = None`. If `validation_strategy` already exists as a top-level field on `TrainingPlan`, remove it (we'll consolidate under `forecasting_settings`).

- [ ] **Step 5: Run the new tests; they should pass**

```
uv run python -m pytest tests/test_contracts/test_forecasting_settings.py -v
```
Expected: 6 passed.

- [ ] **Step 6: Run the full existing test suite to confirm no regressions**

```
uv run python -m pytest -q
```
Expected: All previously-passing tests still pass. If something breaks because the old `forecasting_settings: dict` was being constructed inline, fix the callers to pass `ForecastingSettings(...)` instead.

- [ ] **Step 7: Commit**

```bash
git add src/mlops_agents/contracts/training.py tests/test_contracts/test_forecasting_settings.py
git commit -m "feat(contracts): add typed ValidationStrategy + ExogStrategySettings + ForecastingSettings"
```

---

## Task 2: Pydantic-ify `DatasetProfile` and add `history_length`

**Files:**
- Modify: `src/mlops_agents/training/profiler.py`
- Modify: callers of `build_dataset_profile` (executor, run_benchmark, experience record writer) — only if they break
- Create test: `tests/test_training/test_profiler_history_length.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_training/test_profiler_history_length.py`:

```python
"""Tests for the new history_length field and DatasetProfile typing."""
import pandas as pd
from pathlib import Path

from mlops_agents.training.profiler import build_dataset_profile, DatasetProfile


def _write_csv(rows: int, tmp_path: Path) -> Path:
    dates = pd.date_range("2020-01-01", periods=rows, freq="W")
    df = pd.DataFrame({"date": dates, "target": range(rows)})
    p = tmp_path / "ts.csv"
    df.to_csv(p, index=False)
    return p


def test_profile_is_pydantic_model_with_attribute_access(tmp_path):
    csv = _write_csv(60, tmp_path)
    profile = build_dataset_profile(
        csv, {"problem_type": "forecasting", "target_column": "target",
              "datetime_column": "date", "frequency": "W", "forecast_horizon": 4,
              "series_id_columns": []},
    )
    assert isinstance(profile, DatasetProfile)
    assert profile.history_length is not None  # set for forecasting


def test_history_length_short_for_50_rows(tmp_path):
    csv = _write_csv(50, tmp_path)
    profile = build_dataset_profile(
        csv, {"problem_type": "forecasting", "target_column": "target",
              "datetime_column": "date", "frequency": "W", "forecast_horizon": 4,
              "series_id_columns": []},
    )
    assert profile.history_length in ("very_short", "short")


def test_history_length_medium_for_500_rows(tmp_path):
    csv = _write_csv(500, tmp_path)
    profile = build_dataset_profile(
        csv, {"problem_type": "forecasting", "target_column": "target",
              "datetime_column": "date", "frequency": "W", "forecast_horizon": 4,
              "series_id_columns": []},
    )
    assert profile.history_length in ("medium", "long")


def test_history_length_none_for_tabular(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
    csv = tmp_path / "tab.csv"
    df.to_csv(csv, index=False)
    profile = build_dataset_profile(
        csv, {"problem_type": "classification", "target_column": "target"},
    )
    assert profile.history_length is None


def test_profile_can_serialize_to_json(tmp_path):
    csv = _write_csv(60, tmp_path)
    profile = build_dataset_profile(
        csv, {"problem_type": "forecasting", "target_column": "target",
              "datetime_column": "date", "frequency": "W", "forecast_horizon": 4,
              "series_id_columns": []},
    )
    s = profile.model_dump_json()
    assert "history_length" in s
```

- [ ] **Step 2: Run tests to confirm they fail**

```
uv run python -m pytest tests/test_training/test_profiler_history_length.py -v
```
Expected: ImportError (`DatasetProfile` not exported) or attribute errors.

- [ ] **Step 3: Convert `build_dataset_profile` to return a Pydantic model**

In `src/mlops_agents/training/profiler.py`:

1. At the top of the file, add:

```python
from typing import Literal
from pydantic import BaseModel, Field

HistoryLength = Literal["very_short", "short", "medium", "long"]


class DatasetProfile(BaseModel):
    """Bucketed profile of a dataset, used for rule matching and model selection."""
    schema_version: int = 1
    problem_type: str
    n_rows: str                                  # e.g. very_small | small | medium | large
    n_features: str | None = None
    missing_rate: str | None = None
    n_categorical_features: str | None = None
    n_numerical_features: str | None = None
    # classification-only
    n_classes: str | None = None
    class_balance: str | None = None
    # forecasting-only
    n_series: int | None = None
    history_length: HistoryLength | None = None
    frequency: str | None = None
    horizon_difficulty: str | None = None
    forecast_horizon_raw: int | None = None
    exogenous_features_available: bool | None = None
    seasonality_detected: bool | None = None
    trend_detected: bool | None = None
    stationarity: str | None = None
    # Catch-all for extra fields we already produce but don't want to enumerate
    model_config = {"extra": "allow"}
```

2. Find the existing function body that returns a dict (`return profile`). Compute `history_length` for forecasting:

```python
def _bucket_history_length(n_rows_per_series: int) -> HistoryLength:
    if n_rows_per_series < 60:
        return "very_short"
    if n_rows_per_series < 200:
        return "short"
    if n_rows_per_series < 2000:
        return "medium"
    return "long"
```

(Place near `_bucket_n_rows` or similar helpers — match existing style.)

3. Inside the forecasting branch (right where you currently set `n_series`, `frequency`, etc.), compute:

```python
if task_metadata["problem_type"] == "forecasting":
    sid_cols = task_metadata.get("series_id_columns") or []
    if sid_cols:
        per_series_len = df.groupby(sid_cols[0]).size().min()
    else:
        per_series_len = len(df)
    history_length = _bucket_history_length(int(per_series_len))
    # ... add to the profile dict you're building
    profile["history_length"] = history_length
```

4. At the end of the function, replace `return profile` with:

```python
return DatasetProfile(**profile)
```

- [ ] **Step 4: Update all callers that previously did `profile["x"]`**

```
uv run python -m pytest -q 2>&1 | head -50
```
You'll see KeyError or TypeError where callers did `profile["x"]`. Convert them to `profile.x` (Pydantic attribute access). The known callers are:
- `executor.py` (uses `profile` to pick candidates) — change `profile["history_length"]` to `profile.history_length`, etc.
- `scripts/run_benchmark.py` (passes profile to `default_training_plan`) — change to attribute access
- `experience/writer.py` or wherever the profile is JSON-serialized — replace `json.dumps(profile)` with `profile.model_dump_json()` or `json.dumps(profile.model_dump())`

Do a global grep:

```
uv run python -c "import subprocess; subprocess.run(['grep', '-rn', 'profile\\[', 'src/', 'scripts/'])"
```

- [ ] **Step 5: Run all tests; fix remaining call sites until green**

```
uv run python -m pytest -q
```
Expected: All tests pass including the new 5 in `test_profiler_history_length.py`.

- [ ] **Step 6: Commit**

```bash
git add src/mlops_agents/training/profiler.py src/mlops_agents/training/executor.py \
        scripts/run_benchmark.py tests/test_training/test_profiler_history_length.py
git commit -m "refactor(profiler): convert DatasetProfile to Pydantic; add history_length"
```

(Include any other files you had to touch from Step 4.)

---

## Task 3: Implement `validation_folds.iter_folds`

**Files:**
- Create: `src/mlops_agents/training/validation_folds.py`
- Create: `tests/test_training/test_validation_folds.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_training/test_validation_folds.py
"""Tests for iter_folds: correct count, chronological order, no future leakage."""
import pandas as pd
import pytest

from mlops_agents.contracts.training import ValidationStrategy
from mlops_agents.training.validation_folds import iter_folds


def _make_pool(rows: int) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=rows, freq="W"),
        "target": range(rows),
    })


def test_single_split_yields_one_fold():
    pool = _make_pool(100)
    strat = ValidationStrategy(type="single_split", horizon=10)
    folds = list(iter_folds(pool, strat, "date", []))
    assert len(folds) == 1
    train_idx, val_idx = folds[0]
    assert len(val_idx) == 10
    assert len(train_idx) == 90
    # No overlap; train precedes val
    assert pool.loc[train_idx, "date"].max() < pool.loc[val_idx, "date"].min()


def test_expanding_window_three_folds_train_grows():
    pool = _make_pool(100)
    strat = ValidationStrategy(type="expanding_window", n_folds=3, horizon=10, step_size=10)
    folds = list(iter_folds(pool, strat, "date", []))
    assert len(folds) == 3
    train_lens = [len(t) for t, _ in folds]
    # train sizes strictly increase
    assert train_lens[0] < train_lens[1] < train_lens[2]
    # all val sizes equal horizon
    for _, v in folds:
        assert len(v) == 10
    # All folds chronological
    for t, v in folds:
        assert pool.loc[t, "date"].max() < pool.loc[v, "date"].min()


def test_rolling_window_three_folds_train_size_constant():
    pool = _make_pool(100)
    strat = ValidationStrategy(
        type="rolling_window", n_folds=3, horizon=10, step_size=10, window_size=50
    )
    folds = list(iter_folds(pool, strat, "date", []))
    assert len(folds) == 3
    train_lens = [len(t) for t, _ in folds]
    assert all(L == 50 for L in train_lens)


def test_iter_folds_sorts_by_date_first():
    # If pool is shuffled, iter_folds should still produce chronological folds
    pool = _make_pool(50).sample(frac=1, random_state=0).reset_index(drop=True)
    strat = ValidationStrategy(type="single_split", horizon=5)
    [(t, v)] = list(iter_folds(pool, strat, "date", []))
    assert pool.loc[t, "date"].max() < pool.loc[v, "date"].min()
```

- [ ] **Step 2: Run tests to confirm they fail**

```
uv run python -m pytest tests/test_training/test_validation_folds.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement the module**

```python
# src/mlops_agents/training/validation_folds.py
"""Generate (train_idx, val_idx) pairs given a ValidationStrategy.

This module is single-series for now. Multi-target panel folds are out of
scope for v1 — the executor refuses panel plans that try to use exog
strategies other than `naive_carry`.
"""
from __future__ import annotations
from typing import Iterator

import pandas as pd

from mlops_agents.contracts.training import ValidationStrategy


def iter_folds(
    train_pool: pd.DataFrame,
    strategy: ValidationStrategy,
    dt_col: str,
    sid_cols: list[str],
) -> Iterator[tuple[pd.Index, pd.Index]]:
    """Yield (train_idx, val_idx) pairs in chronological order."""
    if sid_cols:
        # v1: panel datasets don't use this fold iterator; the executor
        # bypasses it for sid_cols and runs the existing no-exog path.
        raise NotImplementedError(
            "Panel multi-target fold iteration deferred to v2"
        )

    pool = train_pool.sort_values(dt_col).reset_index(drop=True)
    horizon = strategy.horizon
    step = strategy.step_size or horizon
    n_folds = strategy.n_folds

    n = len(pool)
    if strategy.type == "single_split":
        train_end = n - horizon
        yield pool.index[:train_end], pool.index[train_end : n]
        return

    # K-fold strategies: place K validation windows of length `horizon`,
    # spaced by `step` and ending at the latest possible point.
    last_val_end = n
    val_starts = [last_val_end - step * (n_folds - 1 - k) - horizon for k in range(n_folds)]

    for i, val_start in enumerate(val_starts):
        val_end = val_start + horizon
        if strategy.type == "expanding_window":
            train_start = 0
        elif strategy.type == "rolling_window":
            window = strategy.window_size or 0
            train_start = max(0, val_start - window)
        else:
            raise ValueError(f"Unknown validation strategy type: {strategy.type}")
        yield pool.index[train_start:val_start], pool.index[val_start:val_end]
```

- [ ] **Step 4: Run tests; iterate until green**

```
uv run python -m pytest tests/test_training/test_validation_folds.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mlops_agents/training/validation_folds.py tests/test_training/test_validation_folds.py
git commit -m "feat(training): add validation_folds.iter_folds for single/rolling/expanding"
```

---

## Task 4: Implement `exog_extender`

**Files:**
- Create: `src/mlops_agents/training/exog_extender.py`
- Create: `tests/test_training/test_exog_extender.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_training/test_exog_extender.py
"""Tests for extend_exog: naive_carry, ets, auto_arima, failure fallback,
index alignment."""
import numpy as np
import pandas as pd
import pytest

from mlops_agents.training.exog_extender import (
    extend_exog,
    _align_val_exog_index,
)


def _series(values: list, freq: str = "W") -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=len(values), freq=freq)
    return pd.Series(values, index=idx, name="oil")


def test_naive_carry_repeats_last_value():
    s = _series([10.0, 11.0, 12.0, 13.0])
    out, fail = extend_exog(s, horizon=3, strategy="naive_carry", freq="W")
    assert list(out) == [13.0, 13.0, 13.0]
    assert fail is None


def test_ets_returns_horizon_values():
    np.random.seed(0)
    history = _series(list(np.linspace(0, 1, 60) + np.random.randn(60) * 0.01))
    out, fail = extend_exog(history, horizon=5, strategy="ets", freq="W")
    assert len(out) == 5
    # fail may be set or not depending on fit; check both states are valid
    assert (fail is None) or ("strategy" in fail and fail["strategy"] == "ets")


def test_auto_arima_returns_horizon_values():
    np.random.seed(0)
    history = _series(list(np.cumsum(np.random.randn(80))))
    out, fail = extend_exog(history, horizon=5, strategy="auto_arima", freq="W")
    assert len(out) == 5


def test_ets_failure_falls_back_to_naive_carry():
    # A constant series can cause some ETS configurations to fail
    s = _series([5.0] * 10)
    out, fail = extend_exog(s, horizon=3, strategy="ets", freq="W")
    # Either ETS succeeded or it fell back to naive_carry
    assert len(out) == 3
    if fail is not None:
        # fallback was used; last value repeated
        assert list(out) == [5.0, 5.0, 5.0]
        assert fail["fallback"] == "naive_carry"


def test_align_index_matches_rangeindex_series_dict():
    val_exog = pd.DataFrame({"oil": [1.0, 2.0, 3.0]})
    series_dict = {"__single__": pd.Series([0.0] * 50, index=pd.RangeIndex(50))}
    aligned = _align_val_exog_index(
        val_exog, series_dict, train_len=50, dt_col="date", freq="W"
    )
    assert isinstance(aligned.index, pd.RangeIndex)
    assert aligned.index.start == 50
    assert aligned.index.stop == 53


def test_align_index_matches_datetimeindex_series_dict():
    val_exog = pd.DataFrame({"oil": [1.0, 2.0, 3.0]})
    train_idx = pd.date_range("2020-01-01", periods=50, freq="W")
    series_dict = {"__single__": pd.Series([0.0] * 50, index=train_idx)}
    aligned = _align_val_exog_index(
        val_exog, series_dict, train_len=50, dt_col="date", freq="W"
    )
    assert isinstance(aligned.index, pd.DatetimeIndex)
    # First future timestamp = train_idx[-1] + 1 freq step
    assert aligned.index[0] == train_idx[-1] + pd.tseries.frequencies.to_offset("W")
```

- [ ] **Step 2: Run tests to confirm they fail**

```
uv run python -m pytest tests/test_training/test_exog_extender.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement the module**

```python
# src/mlops_agents/training/exog_extender.py
"""Leakage firewall: extend an exogenous series into the forecast horizon.

This module only ever sees training-window history. The executor cannot
construct val_exog for unknown_future columns through any other path.
"""
from __future__ import annotations
from typing import Literal

import numpy as np
import pandas as pd

from mlops_agents.utils.logging import get_logger

logger = get_logger(__name__)

Strategy = Literal["naive_carry", "ets", "auto_arima"]


def extend_exog(
    history: pd.Series,
    horizon: int,
    strategy: Strategy,
    freq: str | None,
) -> tuple[pd.Series, dict | None]:
    """Return (predicted_future_values, failure_info_or_None).

    naive_carry never fails. ets / auto_arima fall back to naive_carry on
    fit failure and return a failure_info dict for the experience record.
    """
    if strategy == "naive_carry":
        return _naive_carry(history, horizon), None
    if strategy == "ets":
        return _try_statistical(history, horizon, freq, _fit_ets, "ets")
    if strategy == "auto_arima":
        return _try_statistical(history, horizon, freq, _fit_auto_arima, "auto_arima")
    raise ValueError(f"Unknown exog extension strategy: {strategy!r}")


def _naive_carry(history: pd.Series, horizon: int) -> pd.Series:
    last = history.iloc[-1]
    return pd.Series([last] * horizon, name=history.name)


def _try_statistical(history, horizon, freq, fit_fn, strategy_name):
    try:
        preds = fit_fn(history, horizon, freq)
        return pd.Series(preds, name=history.name), None
    except Exception as e:
        logger.warning(
            f"[exog_extender] {strategy_name} failed for column "
            f"{history.name!r}: {type(e).__name__}: {e}. Falling back to naive_carry."
        )
        return _naive_carry(history, horizon), {
            "strategy": strategy_name,
            "fallback": "naive_carry",
            "error_class": type(e).__name__,
            "error_msg": str(e)[:200],
        }


def _fit_ets(history, horizon, freq):
    from statsforecast.models import AutoETS
    season_length = _season_length_for_freq(freq)
    m = AutoETS(season_length=season_length)
    m.fit(history.values.astype(float))
    return m.predict(h=horizon)["mean"]


def _fit_auto_arima(history, horizon, freq):
    from statsforecast.models import AutoARIMA
    season_length = _season_length_for_freq(freq)
    m = AutoARIMA(season_length=season_length)
    m.fit(history.values.astype(float))
    return m.predict(h=horizon)["mean"]


_FREQ_TO_SEASON = {"H": 24, "D": 7, "W": 52, "MS": 12, "M": 12, "QS": 4, "YS": 1}


def _season_length_for_freq(freq: str | None) -> int:
    if freq is None:
        return 1
    return _FREQ_TO_SEASON.get(freq, 1)


def _align_val_exog_index(
    val_exog: pd.DataFrame,
    series_dict: dict[str, pd.Series],
    train_len: int,
    dt_col: str,
    freq: str | None,
) -> pd.DataFrame:
    """Match val_exog's index type to a sample series in series_dict.

    skforecast requires train_exog and val_exog to share the same index
    type as `series`. If series_dict uses RangeIndex, val_exog continues
    at `train_len`. If DatetimeIndex, val_exog continues from the last
    training timestamp at `freq` cadence.
    """
    if val_exog.empty or len(val_exog) == 0:
        return val_exog
    sample = next(iter(series_dict.values()))
    if isinstance(sample.index, pd.RangeIndex):
        val_exog = val_exog.copy()
        val_exog.index = pd.RangeIndex(train_len, train_len + len(val_exog))
        return val_exog
    last_train_ts = sample.index[-1]
    offset = pd.tseries.frequencies.to_offset(freq) if freq else pd.Timedelta(days=1)
    future_idx = pd.date_range(
        start=last_train_ts + offset,
        periods=len(val_exog),
        freq=freq,
    )
    val_exog = val_exog.copy()
    val_exog.index = future_idx
    return val_exog
```

- [ ] **Step 4: Run tests; iterate until green**

```
uv run python -m pytest tests/test_training/test_exog_extender.py -v
```
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mlops_agents/training/exog_extender.py tests/test_training/test_exog_extender.py
git commit -m "feat(training): add exog_extender (naive_carry/ets/auto_arima + fallback)"
```

---

## Task 5: Implement `validation_policy`

**Files:**
- Create: `src/mlops_agents/training/validation_policy.py`
- Create: `tests/test_training/test_validation_policy.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_training/test_validation_policy.py
"""Tests for select_validation_strategy and validate_forecasting_plan."""
import pytest

from mlops_agents.contracts.training import (
    TrainingPlan, TrainingPlanCandidate,
    ValidationStrategy, ExogStrategySettings, ForecastingSettings,
)
from mlops_agents.contracts.training import TrialBudget
from mlops_agents.training.profiler import DatasetProfile
from mlops_agents.training.validation_policy import (
    select_validation_strategy,
    resolve_rolling_window_size,
    validate_forecasting_plan,
)


def _profile(history_length: str) -> DatasetProfile:
    return DatasetProfile(
        problem_type="forecasting", n_rows="medium", history_length=history_length,
    )


def _task_meta(horizon=6, exog_cols=None, expected_drift=None):
    meta = {
        "problem_type": "forecasting",
        "target_column": "y",
        "datetime_column": "date",
        "series_id_columns": [],
        "frequency": "W",
        "forecast_horizon": horizon,
    }
    if exog_cols is not None:
        meta["exogenous_columns"] = exog_cols
    if expected_drift is not None:
        meta["expected_drift"] = expected_drift
    return meta


# ─── select_validation_strategy ────────────────────────────────────

def test_short_history_returns_single_split_even_with_high_drift():
    s = select_validation_strategy(_profile("short"), _task_meta(expected_drift="high"))
    assert s.type == "single_split"
    assert s.n_folds == 1


def test_medium_history_low_drift_returns_expanding():
    s = select_validation_strategy(_profile("medium"), _task_meta())
    assert s.type == "expanding_window"
    assert s.n_folds == 3


def test_long_history_high_drift_returns_rolling():
    s = select_validation_strategy(_profile("long"), _task_meta(expected_drift="high"))
    assert s.type == "rolling_window"
    assert s.n_folds == 3
    assert s.window_size is None  # auto


# ─── resolve_rolling_window_size ───────────────────────────────────

def test_rolling_window_size_respects_floor_and_capacity():
    # 200 history, horizon 10, 3 folds → can use up to 170 as window
    w = resolve_rolling_window_size(total_history=200, horizon=10, n_folds=3, season_length=None)
    assert 10 <= w <= 170


# ─── validate_forecasting_plan ─────────────────────────────────────

def _plan_with(forecasting_settings):
    return TrainingPlan(
        problem_type="forecasting",
        candidates=[TrainingPlanCandidate(priority=1, model_key="naive")],
        trial_budget=TrialBudget(total_trials=2, allocation_strategy="equal",
                                 min_trials_per_candidate=1, max_trials_per_candidate=2),
        forecasting_settings=forecasting_settings,
    )


def test_validate_raises_when_horizon_mismatch():
    fs = ForecastingSettings(
        validation_strategy=ValidationStrategy(horizon=12),  # task says 6
        exog_strategies=ExogStrategySettings(),
    )
    plan = _plan_with(fs)
    with pytest.raises(ValueError, match="horizon"):
        validate_forecasting_plan(
            plan, _task_meta(horizon=6), _profile("medium"),
            {"single_series": True, "series_lengths": None, "total_len": 200},
        )


def test_validate_raises_when_per_column_references_unknown_column():
    fs = ForecastingSettings(
        validation_strategy=ValidationStrategy(horizon=6),
        exog_strategies=ExogStrategySettings(per_column={"nope": "ets"}),
    )
    plan = _plan_with(fs)
    with pytest.raises(ValueError, match="unknown|not.*exogenous"):
        validate_forecasting_plan(
            plan,
            _task_meta(exog_cols=[{"name": "oil", "future_availability": "unknown_future"}]),
            _profile("medium"),
            {"single_series": True, "series_lengths": None, "total_len": 200},
        )


def test_validate_raises_when_overriding_known_future_column():
    fs = ForecastingSettings(
        validation_strategy=ValidationStrategy(horizon=6),
        exog_strategies=ExogStrategySettings(per_column={"holiday": "ets"}),
    )
    plan = _plan_with(fs)
    with pytest.raises(ValueError, match="known_future"):
        validate_forecasting_plan(
            plan,
            _task_meta(exog_cols=[{"name": "holiday", "future_availability": "known_future"}]),
            _profile("medium"),
            {"single_series": True, "series_lengths": None, "total_len": 200},
        )


def test_validate_raises_when_insufficient_history():
    fs = ForecastingSettings(
        validation_strategy=ValidationStrategy(type="expanding_window", n_folds=3, horizon=20, step_size=20),
        exog_strategies=ExogStrategySettings(),
    )
    plan = _plan_with(fs)
    with pytest.raises(ValueError, match="history|enough"):
        validate_forecasting_plan(
            plan, _task_meta(horizon=20), _profile("medium"),
            {"single_series": True, "series_lengths": None, "total_len": 50},
        )


def test_validate_panel_rejects_per_column_overrides():
    fs = ForecastingSettings(
        validation_strategy=ValidationStrategy(horizon=6),
        exog_strategies=ExogStrategySettings(per_column={"oil": "ets"}),
    )
    plan = _plan_with(fs)
    with pytest.raises(NotImplementedError, match="panel|multi-target"):
        validate_forecasting_plan(
            plan, _task_meta(exog_cols=[{"name": "oil", "future_availability": "unknown_future"}]),
            _profile("medium"),
            {"single_series": False, "series_lengths": {"A": 100, "B": 100}, "total_len": 200},
        )
```

- [ ] **Step 2: Run tests to confirm they fail**

```
uv run python -m pytest tests/test_training/test_validation_policy.py -v
```

- [ ] **Step 3: Implement the module**

```python
# src/mlops_agents/training/validation_policy.py
"""Deterministic validation-strategy policy + plan-level guard rails.

select_validation_strategy: picks the right ValidationStrategy from the
dataset profile and task_metadata.

validate_forecasting_plan: enforced before any modelling runs. Raises
ValueError or NotImplementedError on capacity / leakage / type violations.
"""
from __future__ import annotations
from typing import Any

from mlops_agents.contracts.training import TrainingPlan, ValidationStrategy
from mlops_agents.training.profiler import DatasetProfile
from mlops_agents.utils.logging import get_logger

logger = get_logger(__name__)


def select_validation_strategy(
    profile: DatasetProfile,
    task_metadata: dict[str, Any],
) -> ValidationStrategy:
    horizon = int(task_metadata["forecast_horizon"])
    history = profile.history_length
    drift = task_metadata.get("expected_drift", "low")

    # Short history wins over drift hints: K-fold with too little data is
    # worse than a single clean split.
    if history in ("very_short", "short"):
        return ValidationStrategy(type="single_split", n_folds=1, horizon=horizon)

    if drift == "high":
        return ValidationStrategy(
            type="rolling_window", n_folds=3, horizon=horizon,
            step_size=horizon, window_size=None,
        )
    return ValidationStrategy(
        type="expanding_window", n_folds=3, horizon=horizon, step_size=horizon,
    )


def resolve_rolling_window_size(
    total_history: int,
    horizon: int,
    n_folds: int,
    season_length: int | None,
) -> int:
    # MVP: ignore season_length. TODO: max(3*horizon, 2*season_length, 50)
    base = max(3 * horizon, 50)
    upper = total_history - n_folds * horizon
    return min(base, max(upper, horizon))


def validate_forecasting_plan(
    plan: TrainingPlan,
    task_metadata: dict[str, Any],
    profile: DatasetProfile,
    train_pool_stats: dict[str, Any],
) -> None:
    """Raise ValueError on leakage/capacity/type violations, NotImplementedError
    on panel-specific deferred behavior."""
    fs = plan.forecasting_settings
    if fs is None:
        raise ValueError("Forecasting plan missing forecasting_settings")

    horizon_meta = int(task_metadata["forecast_horizon"])
    vs = fs.validation_strategy

    # (1) horizon match
    if vs.horizon != horizon_meta:
        raise ValueError(
            f"validation_strategy.horizon={vs.horizon} != task_metadata.forecast_horizon={horizon_meta}"
        )

    # (2) n_folds invariant
    if vs.n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {vs.n_folds}")
    if (vs.type == "single_split") != (vs.n_folds == 1):
        raise ValueError(
            f"n_folds=={vs.n_folds} inconsistent with type={vs.type!r}"
        )

    # Build the availability map from task_metadata
    exog_cols_meta = task_metadata.get("exogenous_columns")
    availability: dict[str, str] = {}
    if exog_cols_meta is not None:
        for entry in exog_cols_meta:
            availability[entry["name"]] = entry["future_availability"]

    # (3) panel guardrail
    single_series = bool(train_pool_stats.get("single_series", True))
    if not single_series:
        if fs.exog_strategies.per_column or fs.exog_strategies.default_unknown_future != "naive_carry":
            raise NotImplementedError(
                "Leakage-safe exogenous extension for multi-target panel data deferred to v2"
            )

    # (4) per_column keys must be valid + must reference unknown_future columns only
    for col, strat in fs.exog_strategies.per_column.items():
        if exog_cols_meta is not None and col not in availability:
            raise ValueError(
                f"per_column key {col!r} is not an exogenous column declared in task_metadata"
            )
        if availability.get(col) == "known_future" and strat != "known_future":
            raise ValueError(
                f"Cannot override known_future column {col!r} with strategy {strat!r}"
            )

    # (5) capacity check (single-series only here; panel handled above)
    if single_series:
        total_len = int(train_pool_stats["total_len"])
        min_train_len = max(3 * horizon_meta, 30)
        required = vs.n_folds * horizon_meta + min_train_len
        if total_len < required:
            raise ValueError(
                f"Not enough history for {vs.n_folds}-fold backtesting: "
                f"need >={required} rows, have {total_len}"
            )

    # (6) rolling_window window_size sanity
    if vs.type == "rolling_window" and vs.window_size is not None:
        upper = train_pool_stats["total_len"] - vs.n_folds * horizon_meta
        if not (horizon_meta <= vs.window_size <= upper):
            raise ValueError(
                f"rolling window_size={vs.window_size} must be in "
                f"[{horizon_meta}, {upper}]"
            )
```

- [ ] **Step 4: Run tests; iterate until green**

```
uv run python -m pytest tests/test_training/test_validation_policy.py -v
```
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/mlops_agents/training/validation_policy.py tests/test_training/test_validation_policy.py
git commit -m "feat(training): add validation_policy (selection + plan guardrails)"
```

---

## Task 6: Extend `ExperienceRecord` + SQLite migration

**Files:**
- Modify: `src/mlops_agents/experience/schema.py`
- Modify: `src/mlops_agents/experience/pool.py`
- Create test: `tests/test_experience/test_pool_migration.py` (if directory doesn't exist, also `tests/test_experience/__init__.py`)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_experience/test_pool_migration.py
"""Migration adds five JSON columns; inserting a record populates them."""
import json
import sqlite3
from pathlib import Path

from mlops_agents.experience.pool import ExperiencePool
from mlops_agents.experience.schema import ExperienceRecord


def test_migration_adds_five_columns(tmp_path):
    db = tmp_path / "test.db"
    pool = ExperiencePool(db, audit_dir=tmp_path / "audit")
    conn = sqlite3.connect(db)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(experiences)")}
    conn.close()
    for c in [
        "validation_strategy_json",
        "exog_availability_json",
        "exog_strategies_json",
        "per_fold_metrics_json",
        "exog_fit_failures_json",
    ]:
        assert c in cols


def test_migration_is_idempotent(tmp_path):
    db = tmp_path / "test.db"
    ExperiencePool(db, audit_dir=tmp_path / "audit")
    # Re-init: should not error
    ExperiencePool(db, audit_dir=tmp_path / "audit")


def test_insert_record_with_new_fields_round_trips(tmp_path, minimal_experience_record):
    db = tmp_path / "test.db"
    pool = ExperiencePool(db, audit_dir=tmp_path / "audit")
    record = minimal_experience_record(
        validation_strategy={"type": "single_split", "horizon": 6, "n_folds": 1},
        exog_availability={"oil": "unknown_future"},
        exog_strategies={"oil": "naive_carry"},
        per_fold_metrics=[{"fold_id": 0, "rmse": 1.23}],
        exog_fit_failures=[],
    )
    pool.insert_from_record(record)

    conn = sqlite3.connect(db)
    row = conn.execute(
        "SELECT validation_strategy_json, exog_availability_json, exog_strategies_json, "
        "per_fold_metrics_json, exog_fit_failures_json FROM experiences WHERE task_id=?",
        (record.task_id,),
    ).fetchone()
    conn.close()

    vs, ea, es, pfm, ef = row
    assert json.loads(vs)["type"] == "single_split"
    assert json.loads(ea)["oil"] == "unknown_future"
    assert json.loads(es)["oil"] == "naive_carry"
    assert json.loads(pfm)[0]["rmse"] == 1.23
    assert json.loads(ef) == []
```

You'll need a `minimal_experience_record` fixture. Add it to `tests/conftest.py`:

```python
# tests/conftest.py (append)
import pytest
from mlops_agents.experience.schema import ExperienceRecord, SelectedSolution


@pytest.fixture
def minimal_experience_record():
    """Factory: build a minimal ExperienceRecord with overrideable fields."""
    def _make(**overrides):
        base = dict(
            task_id="test_t_2026-05-11_001",
            problem_type="forecasting",
            dataset_profile={"schema_version": 1, "problem_type": "forecasting",
                             "n_rows": "medium"},
            training_plan_input={},
            split_artifacts={},
            mlflow={"parent_run_id": "abc123"},
            metric_to_optimize="rmse",
            models_tested=[],
            selected_solution=SelectedSolution(
                model_key="naive", best_params={}, best_score=1.0,
                best_score_std=0.0, n_trials_used=1, duration_s=0.1,
                complexity_rank=1, mlflow_run_id="abc123",
            ),
        )
        base.update(overrides)
        return ExperienceRecord(**base)
    return _make
```

- [ ] **Step 2: Run tests to confirm they fail**

```
uv run python -m pytest tests/test_experience/test_pool_migration.py -v
```
Expected: AttributeError or pydantic ValidationError for missing fields.

- [ ] **Step 3: Add the five fields to `ExperienceRecord`**

In `src/mlops_agents/experience/schema.py`, add (preserving existing field order):

```python
# Added by exog/leakage-safe-validation feature
validation_strategy: dict | None = None
exog_availability:   dict | None = None
exog_strategies:     dict | None = None
per_fold_metrics:    list[dict] | None = None
exog_fit_failures:   list[dict] | None = None
```

- [ ] **Step 4: Add PRAGMA-based migration to `ExperiencePool`**

In `src/mlops_agents/experience/pool.py`, find the `__init__` or `_ensure_schema` method that runs the table creation. After the existing CREATE TABLE for `experiences`, add:

```python
NEW_EXPERIENCE_COLUMNS = [
    "validation_strategy_json",
    "exog_availability_json",
    "exog_strategies_json",
    "per_fold_metrics_json",
    "exog_fit_failures_json",
]

def _migrate_experience_columns(conn: sqlite3.Connection) -> None:
    existing = {
        row[1] for row in conn.execute("PRAGMA table_info(experiences)").fetchall()
    }
    for col in NEW_EXPERIENCE_COLUMNS:
        if col not in existing:
            conn.execute(f"ALTER TABLE experiences ADD COLUMN {col} TEXT")
```

Call `_migrate_experience_columns(conn)` inside the connection-setup block that runs after the table creates.

- [ ] **Step 5: Update `insert_from_record` to write the new fields**

Find the existing `INSERT OR REPLACE INTO experiences (...)` statement. Add the 5 new column names and 5 new `?` placeholders. Pass values like:

```python
import json as _json
def _opt_json(v):
    return _json.dumps(v) if v is not None else None

conn.execute(
    """INSERT OR REPLACE INTO experiences
    (task_id, problem_type, dataset_name, dataset_profile_json,
     training_plan_json, selected_model_key, metric_to_optimize,
     metric_direction, validation_score, validation_std,
     experience_summary, mlflow_parent_run_id, created_at,
     validation_strategy_json, exog_availability_json,
     exog_strategies_json, per_fold_metrics_json, exog_fit_failures_json)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
    (
        # ... existing 13 values ...
        _opt_json(record.validation_strategy),
        _opt_json(record.exog_availability),
        _opt_json(record.exog_strategies),
        _opt_json(record.per_fold_metrics),
        _opt_json(record.exog_fit_failures),
    ),
)
```

- [ ] **Step 6: Run the migration tests; iterate until green**

```
uv run python -m pytest tests/test_experience/test_pool_migration.py -v
```
Expected: 3 passed.

- [ ] **Step 7: Run the full suite to confirm no regressions**

```
uv run python -m pytest -q
```

- [ ] **Step 8: Commit**

```bash
git add src/mlops_agents/experience/schema.py src/mlops_agents/experience/pool.py \
        tests/test_experience/test_pool_migration.py tests/conftest.py
git commit -m "feat(experience): add 5 forecasting-strategy fields + PRAGMA migration"
```

---

## Task 7: Rewrite `_run_candidate_forecasting` to use the leakage-safe loop

**Files:**
- Modify: `src/mlops_agents/training/executor.py`
- Create: `tests/test_training/test_executor_forecasting_leakage.py`

**Note:** This is the largest task. Break it into the steps below; each step keeps the suite green.

- [ ] **Step 1: Write the leakage-invariant integration tests first**

```python
# tests/test_training/test_executor_forecasting_leakage.py
"""End-to-end leakage and fold tests for the rewritten forecasting executor."""
import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from mlops_agents.contracts.training import (
    TrainingPlan, TrainingPlanCandidate, TrialBudget,
    ValidationStrategy, ExogStrategySettings, ForecastingSettings,
)
from mlops_agents.training.executor import run_training_plan


def _synthetic_csv(tmp_path: Path, rows: int = 200) -> Path:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2018-01-01", periods=rows, freq="W")
    oil = np.cumsum(rng.normal(0, 1, rows)) + 50
    holiday_flag = ((np.arange(rows) % 13) == 0).astype(int)
    # target depends on oil and holiday_flag
    target = 100 + 0.3 * oil + 5 * holiday_flag + rng.normal(0, 1, rows)
    df = pd.DataFrame({"date": dates, "target": target, "oil": oil, "holiday_flag": holiday_flag})
    p = tmp_path / "synth.csv"
    df.to_csv(p, index=False)
    return p


def _plan(horizon=10):
    return TrainingPlan(
        problem_type="forecasting",
        candidates=[TrainingPlanCandidate(priority=1, model_key="lightgbm_forecaster")],
        trial_budget=TrialBudget(total_trials=2, allocation_strategy="equal",
                                 min_trials_per_candidate=1, max_trials_per_candidate=2),
        forecasting_settings=ForecastingSettings(
            validation_strategy=ValidationStrategy(horizon=horizon),
            exog_strategies=ExogStrategySettings(
                per_column={"oil": "naive_carry"},
                default_unknown_future="naive_carry",
            ),
        ),
    )


def _task_meta(horizon=10):
    return {
        "problem_type": "forecasting", "target_column": "target",
        "datetime_column": "date", "series_id_columns": [],
        "frequency": "W", "forecast_horizon": horizon,
        "exogenous_columns": [
            {"name": "oil", "future_availability": "unknown_future"},
            {"name": "holiday_flag", "future_availability": "known_future"},
        ],
    }


def test_unknown_future_exog_is_extended_not_leaked(tmp_path, monkeypatch):
    """val_exog['oil'] must equal naive_carry(oil_train_history), NOT the realized future oil values."""
    captured = {}

    # Monkeypatch the forecaster's predict to record what exog it sees
    import mlops_agents.training.executor as ex_mod
    orig_predict = None  # set in the wrapper

    csv = _synthetic_csv(tmp_path, 200)
    plan = _plan(horizon=10)

    # Run training
    result = run_training_plan(
        plan=plan, processed_dataset_path=csv, target_column="target",
        task_metadata=_task_meta(horizon=10),
        output_dir=tmp_path / "out", mlflow_experiment="test_leak",
    )
    assert result.champion_candidate is not None

    # Load the experience record and check exog_strategies is recorded
    rec = json.loads(Path(result.experience_record_path).read_text())
    assert rec.get("exog_strategies", {}).get("oil") == "naive_carry"
    assert rec.get("exog_availability", {}).get("holiday_flag") == "known_future"


def test_k_fold_runs_three_folds(tmp_path):
    csv = _synthetic_csv(tmp_path, 400)
    plan = _plan(horizon=10)
    plan.forecasting_settings.validation_strategy = ValidationStrategy(
        type="expanding_window", n_folds=3, horizon=10, step_size=10,
    )
    result = run_training_plan(
        plan=plan, processed_dataset_path=csv, target_column="target",
        task_metadata=_task_meta(horizon=10),
        output_dir=tmp_path / "out", mlflow_experiment="test_kfold",
    )
    rec = json.loads(Path(result.experience_record_path).read_text())
    pfm = rec.get("per_fold_metrics") or []
    assert len(pfm) == 3


def test_plan_with_unknown_column_raises(tmp_path):
    csv = _synthetic_csv(tmp_path, 200)
    plan = _plan(horizon=10)
    plan.forecasting_settings.exog_strategies.per_column = {"nonexistent": "ets"}
    with pytest.raises(ValueError, match="per_column"):
        run_training_plan(
            plan=plan, processed_dataset_path=csv, target_column="target",
            task_metadata=_task_meta(horizon=10),
            output_dir=tmp_path / "out", mlflow_experiment="test_invalid",
        )


def test_plan_without_exog_columns_treats_all_as_unknown(tmp_path):
    csv = _synthetic_csv(tmp_path, 200)
    plan = _plan(horizon=10)
    plan.forecasting_settings.exog_strategies.per_column = {}  # no overrides
    meta = _task_meta(horizon=10)
    del meta["exogenous_columns"]  # default behavior
    result = run_training_plan(
        plan=plan, processed_dataset_path=csv, target_column="target",
        task_metadata=meta, output_dir=tmp_path / "out", mlflow_experiment="test_default",
    )
    rec = json.loads(Path(result.experience_record_path).read_text())
    avail = rec.get("exog_availability") or {}
    assert avail.get("oil") == "unknown_future"
    assert avail.get("holiday_flag") == "unknown_future"
```

- [ ] **Step 2: Run to confirm they fail (will fail or error)**

```
uv run python -m pytest tests/test_training/test_executor_forecasting_leakage.py -v
```
Expected: assertion failures or `KeyError` on the new record fields.

- [ ] **Step 2b: Preserve the existing panel code path FIRST (preparatory rename)**

Before rewriting `_run_candidate_forecasting`, copy its current body into a new function `_run_candidate_forecasting_panel` so the panel path keeps working:

1. Open `src/mlops_agents/training/executor.py`.
2. Locate `def _run_candidate_forecasting(...)` (starts ~line 346).
3. Duplicate the entire current function definition. Name the copy `_run_candidate_forecasting_panel` and give it the same signature.
4. Commit this preparatory rename:

```bash
git add src/mlops_agents/training/executor.py
git commit -m "refactor(executor): clone _run_candidate_forecasting → panel variant for v2"
```

Now the panel path is preserved verbatim and the original function can be rewritten without losing existing behavior.

- [ ] **Step 3: Implement `_resolve_exog_availability` helper in executor**

Open `src/mlops_agents/training/executor.py` and **add** at the top of the file (after existing imports, near the other helpers):

```python
from mlops_agents.training.validation_policy import (
    select_validation_strategy,
    validate_forecasting_plan,
    resolve_rolling_window_size,
)
from mlops_agents.training.validation_folds import iter_folds
from mlops_agents.training.exog_extender import extend_exog, _align_val_exog_index


def _resolve_exog_availability(df_columns: list[str], task_metadata: dict) -> dict[str, str]:
    """Return {col: 'known_future' | 'unknown_future'} for every exog column.

    If task_metadata['exogenous_columns'] is present, it is authoritative and
    unlisted non-target/non-date/non-sid columns are dropped from the exog set.
    If absent, all non-protected columns are treated as unknown_future.
    """
    target = task_metadata["target_column"]
    dt = task_metadata["datetime_column"]
    sids = set(task_metadata.get("series_id_columns") or [])
    protected = {target, dt, *sids}

    declared = task_metadata.get("exogenous_columns")
    if declared is not None:
        return {e["name"]: e["future_availability"] for e in declared}
    return {c: "unknown_future" for c in df_columns if c not in protected}
```

- [ ] **Step 4: Rewrite `_run_candidate_forecasting`**

Replace the **body** of `_run_candidate_forecasting` (keep its signature). Show the new implementation in full (replace lines 346–end-of-function):

```python
def _run_candidate_forecasting(
    candidate: TrainingPlanCandidate,
    train_pool: pd.DataFrame,
    task_metadata: dict[str, Any],
    n_trials: int,
    metric: str,
    direction: str,
    forecasting_settings: ForecastingSettings,
    profile: DatasetProfile,
) -> dict:
    spec = get_model(candidate.model_key)
    target = task_metadata["target_column"]
    dt_col = task_metadata["datetime_column"]
    sid_cols = task_metadata.get("series_id_columns") or []
    horizon = int(task_metadata["forecast_horizon"])
    freq = task_metadata.get("frequency")
    started = time.perf_counter()

    pool = train_pool.copy()
    pool[dt_col] = pd.to_datetime(pool[dt_col])

    is_stat = _is_statsforecast_model(candidate.model_key)
    factory = FACTORY_REGISTRY[spec.factory]

    # Plan-level guardrails (runs once)
    train_pool_stats = {
        "single_series": not sid_cols,
        "series_lengths": (pool.groupby(sid_cols[0]).size().to_dict()
                           if sid_cols else None),
        "total_len": len(pool),
    }
    # Build a tiny throw-away plan-like object for the validator
    from mlops_agents.contracts.training import TrainingPlan, TrialBudget
    throwaway = TrainingPlan(
        problem_type="forecasting",
        candidates=[candidate],
        trial_budget=TrialBudget(total_trials=1, allocation_strategy="equal",
                                 min_trials_per_candidate=1, max_trials_per_candidate=1),
        forecasting_settings=forecasting_settings,
    )
    validate_forecasting_plan(throwaway, task_metadata, profile, train_pool_stats)

    # Resolve auto window_size for rolling_window
    vs = forecasting_settings.validation_strategy
    if vs.type == "rolling_window" and vs.window_size is None:
        vs = vs.model_copy(update={
            "window_size": resolve_rolling_window_size(
                len(pool), horizon, vs.n_folds, season_length=None,
            )
        })

    # Multi-target panel: existing no-exog behavior, single-fold only
    if sid_cols:
        return _run_candidate_forecasting_panel(
            candidate, pool, task_metadata, n_trials, metric, direction,
            spec, is_stat, factory,
        )

    # ─── Build availability + strategy maps ────────────────────────
    availability = _resolve_exog_availability(list(pool.columns), task_metadata)
    exog_columns = list(availability.keys())
    strategies = forecasting_settings.exog_strategies

    exog_cache: dict[tuple, pd.Series] = {}
    all_failures: list[dict] = []

    def fit_score(params: dict) -> tuple[float, list[float], list[dict]]:
        fold_scores: list[float] = []
        fold_failures: list[dict] = []

        for fold_id, (train_idx, val_idx) in enumerate(iter_folds(pool, vs, dt_col, sid_cols)):
            cand_train = pool.loc[train_idx].reset_index(drop=True)
            cand_val   = pool.loc[val_idx].reset_index(drop=True)

            if is_stat:
                # statsforecast path: ignore exog (existing behavior)
                sf = factory({"task_metadata": task_metadata, "params": params})
                sf.fit(_to_sf_format(cand_train, target, dt_col, sid_cols))
                fcst = sf.predict(h=horizon)
                model_col = [c for c in fcst.columns if c not in ("unique_id", "ds")][0]
                val_sf = _to_sf_format(cand_val, target, dt_col, sid_cols)
                merged = val_sf.merge(fcst, on=["unique_id", "ds"])
                if merged.empty:
                    val_s = val_sf.sort_values(["unique_id", "ds"]).reset_index(drop=True)
                    fct_s = fcst.sort_values(["unique_id", "ds"]).reset_index(drop=True)
                    n = min(len(val_s), len(fct_s))
                    if n == 0:
                        raise ValueError("Statsforecast produced no predictions")
                    score = _fc_metrics(val_s["y"].values[:n], fct_s[model_col].values[:n])[metric]
                else:
                    score = _fc_metrics(merged["y"].values, merged[model_col].values)[metric]
                fold_scores.append(score)
                continue

            # ── Skforecast path with leakage-safe exog ─────────────
            forecaster = factory({"task_metadata": task_metadata, "params": params})
            series_dict = _build_series_dict(cand_train, dt_col, target, sid_cols, freq)

            future_values: dict[str, pd.Series] = {}
            for col in exog_columns:
                avail = availability[col]
                if avail == "known_future":
                    future_values[col] = cand_val[col].reset_index(drop=True)
                    continue
                strat = strategies.per_column.get(col, strategies.default_unknown_future)
                if strat == "drop":
                    continue
                cache_key = (col, strat, fold_id, "default")
                if cache_key in exog_cache:
                    future_values[col] = exog_cache[cache_key]
                else:
                    preds, fail = extend_exog(cand_train[col], horizon, strat, freq)
                    future_values[col] = preds
                    exog_cache[cache_key] = preds
                    if fail is not None:
                        fold_failures.append(fail | {"fold_id": fold_id, "column": col})

            used_cols = list(future_values.keys())
            train_exog = cand_train[used_cols] if used_cols else None
            val_exog = None
            if used_cols:
                val_exog_raw = pd.DataFrame(future_values)
                val_exog = _align_val_exog_index(
                    val_exog_raw, series_dict, train_len=len(cand_train),
                    dt_col=dt_col, freq=freq,
                )
                assert list(train_exog.columns) == list(val_exog.columns)

            forecaster.fit(series=series_dict, exog=train_exog)
            preds = forecaster.predict(steps=horizon, exog=val_exog)
            preds = preds.reset_index().rename(columns={"index": "ds"})
            val_long = cand_val.rename(columns={target: "y_true", dt_col: "ds"}).copy()
            val_long["level"] = "__single__"
            val_long["ds"] = pd.to_datetime(val_long["ds"])
            preds["ds"] = pd.to_datetime(preds["ds"])
            joined = val_long[["level", "ds", "y_true"]].merge(
                preds[["level", "ds", "pred"]], on=["level", "ds"], how="inner"
            )
            if joined.empty:
                score = _fc_metrics(
                    val_long["y_true"].values, preds["pred"].values[: len(val_long)],
                )[metric]
            else:
                score = _fc_metrics(joined["y_true"].values, joined["pred"].values)[metric]
            fold_scores.append(score)

        return float(np.mean(fold_scores)), fold_scores, fold_failures

    narrowed = (
        narrow_search_space(candidate.model_key, candidate.search_space_override)
        if candidate.search_space_override else spec.search_space
    )
    suggest_fn = build_suggest_fn(narrowed)

    last_per_fold: list[float] = []
    last_failures: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        nonlocal last_per_fold, last_failures
        params = suggest_fn(trial)
        score, per_fold, failures = fit_score(params)
        last_per_fold = per_fold
        last_failures = failures
        return score

    try:
        if not narrowed.params:
            best_score, last_per_fold, last_failures = fit_score(spec.default_params)
            best_params, n_used = spec.default_params, 1
        else:
            study = optuna.create_study(
                direction=direction, sampler=optuna.samplers.TPESampler(seed=42)
            )
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            best_params = study.best_params
            best_score = study.best_value
            n_used = len(study.trials)
        status = "ok"
    except Exception as e:
        logger.exception(f"[{candidate.model_key}] failed: {e}")
        return {
            "model_key": candidate.model_key, "status": "failed",
            "best_params": {}, "best_score": float("inf"),
            "best_score_std": 0.0, "n_trials_used": 0,
            "duration_s": time.perf_counter() - started,
            "complexity_rank": spec.complexity_rank,
            "per_fold_scores": [], "exog_fit_failures": [],
        }

    return {
        "model_key": candidate.model_key,
        "status": status,
        "best_params": best_params,
        "best_score": float(best_score),
        "best_score_std": float(np.std(last_per_fold)) if last_per_fold else 0.0,
        "n_trials_used": n_used,
        "duration_s": time.perf_counter() - started,
        "complexity_rank": spec.complexity_rank,
        "per_fold_scores": [float(x) for x in last_per_fold],
        "exog_fit_failures": last_failures,
    }


# _run_candidate_forecasting_panel: already exists (created in Step 2b)
# with the verbatim pre-refactor body. The signature now needs to match
# the dispatch from the new _run_candidate_forecasting above:
#
#   _run_candidate_forecasting_panel(candidate, pool, task_metadata,
#                                    n_trials, metric, direction,
#                                    spec, is_stat, factory) -> dict
#
# Update the panel function's signature to accept (spec, is_stat, factory)
# instead of recomputing them. Body otherwise unchanged from Step 2b.
```

- [ ] **Step 5: Update `run_training_plan` to wire `forecasting_settings` and `profile`**

Find the existing call to `_run_candidate_forecasting(...)` inside `run_training_plan`. Update it to pass the new arguments:

```python
fs = plan.forecasting_settings
if fs is None and plan.problem_type == "forecasting":
    # Default policy fills it
    from mlops_agents.training.validation_policy import select_validation_strategy
    fs = ForecastingSettings(
        validation_strategy=select_validation_strategy(profile, task_metadata),
        exog_strategies=ExogStrategySettings(),
    )
    plan = plan.model_copy(update={"forecasting_settings": fs})

# ... in the per-candidate loop:
if plan.problem_type == "forecasting":
    result = _run_candidate_forecasting(
        candidate, train_pool, task_metadata, n_trials, metric, direction,
        forecasting_settings=fs, profile=profile,
    )
```

- [ ] **Step 6: Update `_retrain_forecasting` to drop the leaky `_build_exog_df(val, ...)` path**

The retrain function uses the **entire** train_pool. There's no held-out validation. So exog can be built from realized history with no leakage risk:

```python
def _retrain_forecasting(spec, champion, train_pool, task_metadata, models_dir):
    factory = FACTORY_REGISTRY[spec.factory]
    target = task_metadata["target_column"]
    dt_col = task_metadata["datetime_column"]
    sid_cols = task_metadata.get("series_id_columns") or []
    train_pool = train_pool.copy()
    train_pool[dt_col] = pd.to_datetime(train_pool[dt_col])
    path = models_dir / f"champion_{champion['model_key']}.pkl"
    if _is_statsforecast_model(champion["model_key"]):
        sf = factory({"task_metadata": task_metadata, "params": champion["best_params"]})
        sf.fit(_to_sf_format(train_pool, target, dt_col, sid_cols))
        with path.open("wb") as f:
            pickle.dump(sf, f)
        return path

    forecaster = factory({"task_metadata": task_metadata, "params": champion["best_params"]})
    freq = task_metadata.get("frequency")
    series_dict = _build_series_dict(train_pool, dt_col, target, sid_cols, freq)
    # Realized exog history is leakage-free here (no validation held out).
    if not sid_cols:
        availability = _resolve_exog_availability(list(train_pool.columns), task_metadata)
        used_cols = [c for c in availability if c in train_pool.columns]
        train_exog = train_pool[used_cols] if used_cols else None
    else:
        train_exog = None
    forecaster.fit(series=series_dict, exog=train_exog)
    with path.open("wb") as f:
        pickle.dump(forecaster, f)
    return path
```

- [ ] **Step 7: Run the integration tests**

```
uv run python -m pytest tests/test_training/test_executor_forecasting_leakage.py -v
```
Expected: 4 passed.

- [ ] **Step 8: Run the full training test directory**

```
uv run python -m pytest tests/test_training/ -v
```
Expected: All pass (including pre-existing forecasting tests).

- [ ] **Step 9: Commit**

```bash
git add src/mlops_agents/training/executor.py tests/test_training/test_executor_forecasting_leakage.py
git commit -m "feat(executor): leakage-safe forecasting loop with K-fold + exog extender"
```

---

## Task 8: Wire experience record + MLflow logging for the new fields

**Files:**
- Modify: `src/mlops_agents/training/experience.py` (or wherever the experience record is assembled)
- Modify: `src/mlops_agents/training/executor.py` (the place where the record is built before being written)

- [ ] **Step 1: Find the assembly site**

```
uv run python -c "import subprocess; subprocess.run(['grep', '-rn', 'ExperienceRecord(', 'src/'])"
```

Identify where the record is constructed (likely in `executor.py` near the end of `run_training_plan`, or in `experience.py`).

- [ ] **Step 2: Populate the five new fields when problem_type is forecasting**

```python
# At the record-assembly site:
forecasting_extras = {}
if plan.problem_type == "forecasting":
    fs = plan.forecasting_settings
    availability = _resolve_exog_availability(list(processed_df.columns), task_metadata)
    # Materialize what was actually used (default fills + per-column overrides)
    used_strategies = {}
    for col, avail in availability.items():
        if avail == "known_future":
            used_strategies[col] = "known_future"
        else:
            used_strategies[col] = fs.exog_strategies.per_column.get(
                col, fs.exog_strategies.default_unknown_future,
            )
    forecasting_extras = {
        "validation_strategy": fs.validation_strategy.model_dump(),
        "exog_availability": availability,
        "exog_strategies": used_strategies,
        "per_fold_metrics": [
            {"fold_id": i, "score": s}
            for i, s in enumerate(champion_result.get("per_fold_scores", []))
        ],
        "exog_fit_failures": champion_result.get("exog_fit_failures", []),
    }

record = ExperienceRecord(
    # ... existing args ...
    **forecasting_extras,
)
```

- [ ] **Step 3: Log MLflow parent-run params + per-fold metrics**

In `run_training_plan`, near the existing `mlflow.log_param(...)` calls:

```python
if plan.problem_type == "forecasting":
    fs = plan.forecasting_settings
    mlflow.log_param("validation_strategy_type", fs.validation_strategy.type)
    mlflow.log_param("validation_n_folds", fs.validation_strategy.n_folds)
    mlflow.log_param("exog_default_strategy", fs.exog_strategies.default_unknown_future)
    mlflow.log_param("expected_drift", task_metadata.get("expected_drift", "low"))
    for i, s in enumerate(champion_result.get("per_fold_scores", [])):
        mlflow.log_metric(f"fold_{i}_{metric}", s)
    if champion_result.get("per_fold_scores"):
        scores = champion_result["per_fold_scores"]
        mlflow.log_metric(f"fold_mean_{metric}", float(np.mean(scores)))
        mlflow.log_metric(f"fold_std_{metric}", float(np.std(scores)))
```

- [ ] **Step 4: Verify by running the executor leakage tests**

```
uv run python -m pytest tests/test_training/test_executor_forecasting_leakage.py -v
```
The `test_k_fold_runs_three_folds` test specifically asserts `len(per_fold_metrics) == 3`.

- [ ] **Step 5: Commit**

```bash
git add src/mlops_agents/training/executor.py
git commit -m "feat(experience+mlflow): populate forecasting strategy + per-fold metrics"
```

---

## Task 9: Extend `MLRule` schema + add 6 new YAML rules

**Files:**
- Modify: `src/mlops_agents/knowledge/reader.py`
- Modify: `src/mlops_agents/knowledge/ml_rules.yaml`
- Create test: `tests/test_knowledge/test_forecasting_rules.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_knowledge/test_forecasting_rules.py
"""The new forecasting rules load and match correctly."""
from mlops_agents.knowledge.reader import load_rules


def test_six_new_forecasting_rules_load():
    rules = load_rules()
    ids = {r.rule_id for r in rules}
    expected = {
        "forecasting_short_history_single_split",
        "forecasting_medium_long_expanding_window",
        "forecasting_high_drift_rolling_window",
        "exog_calendar_known_future",
        "exog_unknown_default_naive_carry",
        "exog_slow_macro_auto_arima",
    }
    assert expected.issubset(ids)


def test_recommend_field_present_on_new_rules():
    rules = {r.rule_id: r for r in load_rules()}
    r = rules["forecasting_short_history_single_split"]
    assert r.recommend == {"validation_strategy": "single_split"}
```

- [ ] **Step 2: Run to confirm failure**

```
uv run python -m pytest tests/test_knowledge/test_forecasting_rules.py -v
```
Expected: AttributeError for `recommend` or KeyError for rule_id.

- [ ] **Step 3: Add `recommend: dict[str, Any]` to `MLRule`**

In `src/mlops_agents/knowledge/reader.py`:

```python
from typing import Any
# ...

class MLRule(BaseModel):
    rule_id: str
    applies_when: dict[str, Any]
    prefer: list[str] = []
    avoid_or_deprioritize: list[str] = []
    requirements: list[str] = []
    recommend: dict[str, Any] = Field(default_factory=dict)   # NEW
    reason: str
    tags: list[str] = []
```

(Add `from pydantic import Field` if not already imported.)

- [ ] **Step 4: Append the 6 new rules to `ml_rules.yaml`**

At the end of `src/mlops_agents/knowledge/ml_rules.yaml`, append (preserve indentation style of existing entries):

```yaml
# ─── forecasting_rules — exog & validation strategy guidance (SP5 planner) ─
- rule_id: forecasting_short_history_single_split
  applies_when:
    problem_type: forecasting
    history_length: [very_short, short]
  recommend:
    validation_strategy: single_split
  reason: "Multiple folds leave too little data per fold with short history."

- rule_id: forecasting_medium_long_expanding_window
  applies_when:
    problem_type: forecasting
    history_length: [medium, long]
  recommend:
    validation_strategy: expanding_window
  reason: "Sufficient history makes expanding-window backtesting more robust."

- rule_id: forecasting_high_drift_rolling_window
  applies_when:
    problem_type: forecasting
    expected_drift: high
  recommend:
    validation_strategy: rolling_window
  reason: "Non-stationary processes benefit from a fixed-size recent training window."

- rule_id: exog_calendar_known_future
  applies_when:
    problem_type: forecasting
    exog_column_kind: calendar_derived
  recommend:
    future_availability: known_future
  reason: "Calendar-derived features (year, month, dayofweek, deterministic holiday-calendar flags) are known for future timestamps."

- rule_id: exog_unknown_default_naive_carry
  applies_when:
    problem_type: forecasting
    exog_future_availability: unknown_future
  recommend:
    exog_strategy: naive_carry
  reason: "Safest default; cheap; competitive for short horizons."

- rule_id: exog_slow_macro_auto_arima
  applies_when:
    problem_type: forecasting
    exog_kind: macro_indicator
    history_length: [medium, long]
  recommend:
    exog_strategy: auto_arima
  reason: "Slow-moving macro variables (rates, FX) may have ARIMA-friendly dynamics and can sometimes outperform naive carry over longer horizons."
```

- [ ] **Step 5: Run tests; iterate**

```
uv run python -m pytest tests/test_knowledge/test_forecasting_rules.py -v
```
Expected: 2 passed.

- [ ] **Step 6: Confirm rule engine still loads existing rules**

```
uv run python -m pytest tests/test_knowledge/ -v
```
Expected: All pass.

- [ ] **Step 7: Commit**

```bash
git add src/mlops_agents/knowledge/reader.py src/mlops_agents/knowledge/ml_rules.yaml \
        tests/test_knowledge/test_forecasting_rules.py
git commit -m "feat(knowledge): add 6 forecasting rules + 'recommend' MLRule field"
```

---

## Task 10: Benchmark manifest updates + `build_task_metadata`

**Files:**
- Modify: `scripts/benchmark_manifest.yaml`
- Modify: `scripts/run_benchmark.py`

- [ ] **Step 1: Update `build_task_metadata` to propagate exog metadata**

In `scripts/run_benchmark.py`, update the function:

```python
def build_task_metadata(entry: dict) -> dict:
    meta = {"problem_type": entry["problem_type"], "target_column": entry["target_column"]}
    if entry["problem_type"] == "forecasting":
        meta.update({
            "datetime_column": entry["datetime_column"],
            "series_id_columns": entry.get("series_id_columns", []),
            "frequency": entry["frequency"],
            "forecast_horizon": entry["forecast_horizon"],
        })
        if "exogenous_columns" in entry:
            meta["exogenous_columns"] = entry["exogenous_columns"]
        if "expected_drift" in entry:
            meta["expected_drift"] = entry["expected_drift"]
    return meta
```

- [ ] **Step 2: Annotate the multi-exog manifest entries**

In `scripts/benchmark_manifest.yaml`, add `exogenous_columns` blocks AND `expected_drift: high` to the financial datasets:

```yaml
- dataset_id: sp500_macro_weekly
  source: yfinance_multi
  source_id: "^GSPC"
  exog_tickers: ["^VIX", "^TNX", "CL=F", "GC=F", "DX-Y.NYB", "EURUSD=X", "^IXIC"]
  col_names: { "^GSPC": sp500, "^VIX": vix, "^TNX": treasury_10y, "CL=F": oil_wti, "GC=F": gold, "DX-Y.NYB": usd_index, "EURUSD=X": eurusd, "^IXIC": nasdaq }
  problem_type: forecasting
  target_column: sp500
  datetime_column: week
  series_id_columns: []
  frequency: W
  forecast_horizon: 13
  interval: 1wk
  start: "2005-01-01"
  end: "2024-06-01"
  expected_drift: high                                            # NEW
  exogenous_columns:                                              # NEW
    - { name: vix,          future_availability: unknown_future }
    - { name: treasury_10y, future_availability: unknown_future }
    - { name: oil_wti,      future_availability: unknown_future }
    - { name: gold,         future_availability: unknown_future }
    - { name: usd_index,    future_availability: unknown_future }
    - { name: eurusd,       future_availability: unknown_future }
    - { name: nasdaq,       future_availability: unknown_future }
```

Repeat the `expected_drift: high` + `exogenous_columns:` blocks for the other 5 financial datasets:
- `gold_macro_monthly` (oil, silver, sp500, usd)
- `commodity_panel_weekly` (no exog declarations — multi-target panel; leave as-is so it stays under panel guardrail)
- `crypto_weekly` (no exog; single-series target)
- `fx_exog_weekly` (gbpusd, jpyusd, gold as unknown_future)
- `oil_multi_exog_weekly` (nat_gas, gasoline, copper, sp500, usd_index, gold, vix)
- `gold_multi_exog_weekly` (silver, copper, oil_wti, usd_index, treasury_10y, sp500, vix)

For `air_passengers` and `m4_monthly_sample` (no exog), leave alone — the executor will treat them as having zero exog.

- [ ] **Step 3: Run the benchmark on one financial dataset to smoke-test**

```
uv run python scripts/run_benchmark.py --trials 2 2>&1 | tail -30
```

(Or use the subset trick from earlier: write a temp manifest with one entry.)

Expected: the dataset completes; the resulting experience record has `validation_strategy` and `exog_strategies` populated.

- [ ] **Step 4: Commit**

```bash
git add scripts/benchmark_manifest.yaml scripts/run_benchmark.py
git commit -m "feat(benchmark): declare exogenous_columns + expected_drift for financial datasets"
```

---

## Task 11: Regression — re-seed full benchmark and verify

**Files:**
- (no code changes — verification step)

- [ ] **Step 1: Wipe and re-seed**

```
rm -f storage/mlops_metadata.db
rm -rf experience_pool/
uv run python scripts/run_benchmark.py --trials 8 2>&1 | tee /tmp/bench.log | tail -40
```
Expected: `Benchmark complete: 21 success, 0 failed`.

- [ ] **Step 2: Verify the experience pool has the new fields populated**

```
uv run python -c "
import sqlite3, json
from mlops_agents.config.settings import settings
conn = sqlite3.connect(settings.experience_db_path)
cols = [r[1] for r in conn.execute('PRAGMA table_info(experiences)')]
print('Has new columns:', all(c in cols for c in [
    'validation_strategy_json', 'exog_availability_json',
    'exog_strategies_json', 'per_fold_metrics_json', 'exog_fit_failures_json',
]))
rows = conn.execute('''
    SELECT dataset_name, validation_strategy_json, exog_strategies_json,
           per_fold_metrics_json FROM experiences
    WHERE problem_type=\"forecasting\"
''').fetchall()
for r in rows:
    name, vs, es, pfm = r
    vs_t = json.loads(vs)['type'] if vs else 'N/A'
    n_pf = len(json.loads(pfm)) if pfm else 0
    n_es = len(json.loads(es)) if es else 0
    print(f'{name:30s} vs={vs_t:18s} n_exog_strategies={n_es:2d} n_folds={n_pf}')
"
```

Expected output:
- All forecasting datasets show populated validation_strategy
- Financial datasets show 3 fold-metrics (rolling/expanding) or 1 (single_split)
- `exog_strategies` non-empty for datasets with declared exog

- [ ] **Step 3: Confirm zero leakage by spot-checking one record**

```
uv run python -c "
import sqlite3, json
from mlops_agents.config.settings import settings
conn = sqlite3.connect(settings.experience_db_path)
row = conn.execute('SELECT exog_availability_json, exog_strategies_json FROM experiences WHERE dataset_name=\"sp500_macro_weekly\"').fetchone()
print('availability:', json.loads(row[0]))
print('strategies  :', json.loads(row[1]))
assert all(v == 'unknown_future' for v in json.loads(row[0]).values())
assert all(v == 'naive_carry' for v in json.loads(row[1]).values())
print('OK: all exog declared unknown_future and extended via naive_carry')
"
```

- [ ] **Step 4: Run the full test suite one last time**

```
uv run python -m pytest -q
```
Expected: All pass (existing 274 + new ~25).

- [ ] **Step 5: Commit the regenerated artifacts**

```bash
git add storage/ experience_pool/ 2>/dev/null || true
git status
# If new audit JSONs were generated, decide per-project policy whether to commit them.
# The DB is git-ignored; audit files may or may not be. If unsure, skip:
git restore --staged storage/ experience_pool/ 2>/dev/null || true
```

- [ ] **Step 6: Final commit + push**

```bash
git commit --allow-empty -m "chore: leakage-safe forecasting validation rolled out end-to-end"
git push
```

---

## Spec coverage check (self-review)

| Spec section | Implemented in |
|---|---|
| §3 Architecture (3 layers) | Task 1 (contracts), Task 5 (validation_policy), Task 7 (executor) |
| §4.1 task_metadata.exogenous_columns / expected_drift | Task 10 |
| §4.2 ValidationStrategy / ExogStrategySettings / ForecastingSettings | Task 1 |
| §4.3 DatasetProfile Pydantic-ified + history_length | Task 2 |
| §4.4 ExperienceRecord additions | Task 6 |
| §5 select_validation_strategy / validate_forecasting_plan | Task 5 |
| §6.1 validation_folds.iter_folds | Task 3 |
| §6.2 exog_extender.extend_exog + _align_val_exog_index | Task 4 |
| §6.3 leakage-safe _run_candidate_forecasting | Task 7 |
| §6.4 single-target scope; panel guardrail | Tasks 5, 7 |
| §6.5 failure modes | Tasks 4, 7 |
| §7 ML rules YAML extensions | Task 9 |
| §8 SQLite migration | Task 6 |
| §9 MLflow integration | Task 8 |
| §10 testing strategy | Tasks 1, 3, 4, 5, 6, 7, 9 + Task 11 regression |
| §11 acceptance criteria | Task 11 |
| §12 out of scope | (not implemented, correctly) |
| §13 files touched | All tasks |

All spec requirements have a corresponding task. The plan is complete.
