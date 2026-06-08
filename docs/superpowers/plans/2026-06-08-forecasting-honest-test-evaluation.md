# Forecasting Honest Test-Set Evaluation (SP1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Project commit policy:** CLAUDE.md says "Never commit changes" and "Never add Claude as co-author." Commit steps are included for structure, but follow the user's policy — do not add any `Co-Authored-By: Claude` trailer, and only commit if the user has authorized it for this execution.

**Goal:** Make forecasting runs report and chart the champion's honest **held-out test** performance (not the validation window), surface the validation score separately, and make exog handling consistent by removing the `drop` strategy.

**Architecture:** A new, unit-tested helper `_forecast_champion_on_test` forecasts the train-pool-retrained champion across the held-out test horizon (statsforecast → `predict(h)`; skforecast → extend unknown-future exog from train history, use actual known-future values, then `predict(steps, exog)`), returning test metrics + a chart preview. The executor's forecasting branch reports test metrics as `champion_metrics`, keeps the validation score as `selection_score`, logs test as the primary MLflow `rmse` and validation as `selection_{metric}`, and builds the chart from the test preview. The `drop` exog strategy is removed everywhere so validation, retrain, and test use the same feature set.

**Tech Stack:** Python 3.12, pandas, numpy, statsforecast, skforecast, Optuna, MLflow, Pydantic, FastAPI, Next.js/React; UV + pytest.

**Spec:** `docs/superpowers/specs/2026-06-08-forecasting-honest-test-evaluation-design.md`

---

## File Structure

| File | Responsibility | Change |
|------|----------------|--------|
| `src/mlops_agents/contracts/training.py` | Exog strategy Literals; `TrainingResult` | remove `"drop"`; add `selection_score` |
| `src/mlops_agents/planning/validation.py` | Planner output validation | remove `"drop"` from allowed exog strategies; reword message |
| `src/mlops_agents/prompts/planner.yaml` | Planner system prompt | remove `drop` from strategy options |
| `src/mlops_agents/contracts/outputs.py` | `TrainingStateUpdate` | add `selection_score` + map in `from_training_result` |
| `src/mlops_agents/state/agent_state.py` | `AgentState` | add `selection_score` key |
| `src/mlops_agents/training/executor.py` | training executor | new helpers `_build_test_exog`, `_forecast_champion_on_test`; remove `drop` branch + `_val_preview` capture in `fit_score`; rewire forecasting metrics branch; MLflow logging |
| `api/services/pipeline.py` | SSE event builder | add `selection_score` to `training_complete` event |
| `frontend/components/pipeline/ResultsDashboard.tsx` | Model tab panel | secondary validation line; chart legend label; test-unavailable case; types |
| `tests/test_contracts/test_training.py` | contract tests | `drop` rejected; `selection_score` field |
| `tests/test_training/test_forecast_champion_on_test.py` | new helper tests | statsforecast happy path; no-leakage exog; graceful guard |
| `tests/test_training/test_executor_forecasting.py` | executor tests | test-eval semantics; `drop` removal |

---

## Task 1: Remove the `drop` exog strategy (contracts + planner)

**Files:**
- Modify: `src/mlops_agents/contracts/training.py:18-19`
- Modify: `src/mlops_agents/planning/validation.py:24,110-114`
- Modify: `src/mlops_agents/prompts/planner.yaml:48`
- Test: `tests/test_contracts/test_training.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_contracts/test_training.py`:

```python
import pytest
from pydantic import ValidationError
from mlops_agents.contracts.training import ExogStrategySettings


def test_drop_strategy_is_rejected_default():
    with pytest.raises(ValidationError):
        ExogStrategySettings(default_unknown_future="drop")


def test_drop_strategy_is_rejected_per_column():
    with pytest.raises(ValidationError):
        ExogStrategySettings(per_column={"temp": "drop"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_contracts/test_training.py -k drop -v`
Expected: FAIL — `"drop"` is currently a valid Literal, so no `ValidationError` is raised.

- [ ] **Step 3: Remove `drop` from the Literals**

In `src/mlops_agents/contracts/training.py`, change lines 18-19 from:

```python
ExogStrategy = Literal["known_future", "naive_carry", "ets", "auto_arima", "drop"]
UnknownFutureStrategy = Literal["naive_carry", "ets", "auto_arima", "drop"]
```

to:

```python
ExogStrategy = Literal["known_future", "naive_carry", "ets", "auto_arima"]
UnknownFutureStrategy = Literal["naive_carry", "ets", "auto_arima"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_contracts/test_training.py -k drop -v`
Expected: PASS

- [ ] **Step 5: Remove `drop` from planner validation**

In `src/mlops_agents/planning/validation.py`, change line 24 from:

```python
ALLOWED_EXOG_STRATEGIES = {"naive_carry", "ets", "auto_arima", "drop"}
```

to:

```python
ALLOWED_EXOG_STRATEGIES = {"naive_carry", "ets", "auto_arima"}
```

And reword the message at lines 110-114 from:

```python
            if col in known_future:
                raise PlannerValidationError(
                    f"known_future column {col!r} cannot appear in per-column "
                    f"unknown-future strategies (no 'drop' loophole)"
                )
```

to:

```python
            if col in known_future:
                raise PlannerValidationError(
                    f"known_future column {col!r} cannot appear in per-column "
                    f"unknown-future strategies"
                )
```

- [ ] **Step 6: Remove `drop` from the planner prompt**

In `src/mlops_agents/prompts/planner.yaml`, line 48 currently reads (in context of describing exog strategies):

```yaml
    {naive_carry, ets, auto_arima, drop}.
```

Change to:

```yaml
    {naive_carry, ets, auto_arima}.
```

- [ ] **Step 7: Run the contract + planner tests**

Run: `uv run pytest tests/test_contracts/ tests/test_agents/test_planner_node.py -q -m "not integration"`
Expected: PASS (no test referenced the `drop` string, so none should break).

- [ ] **Step 8: Commit**

```bash
git add src/mlops_agents/contracts/training.py src/mlops_agents/planning/validation.py src/mlops_agents/prompts/planner.yaml tests/test_contracts/test_training.py
git commit -m "feat(forecasting): remove drop exog strategy from contracts and planner"
```

---

## Task 2: Add `selection_score` to contracts and state

**Files:**
- Modify: `src/mlops_agents/contracts/training.py` (`TrainingResult`)
- Modify: `src/mlops_agents/contracts/outputs.py` (`TrainingStateUpdate`)
- Modify: `src/mlops_agents/state/agent_state.py` (`AgentState`)
- Test: `tests/test_contracts/test_training.py`, existing `tests/test_contracts/test_state_binding.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_contracts/test_training.py`:

```python
from mlops_agents.contracts.training import TrainingResult


def test_training_result_has_selection_score_default_none():
    r = TrainingResult(
        champion_candidate={"model_key": "ets"},
        champion_model_path="x.pkl",
        train_pool_path="t.csv",
        test_path="te.csv",
        split_metadata_path="s.json",
        mlflow_parent_run_id="abc",
        experience_record_path="e.json",
        champion_metrics={"rmse": 1.0},
    )
    assert r.selection_score is None
    r2 = r.model_copy(update={"selection_score": 4.74})
    assert r2.selection_score == 4.74
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_contracts/test_training.py -k selection_score -v`
Expected: FAIL — `selection_score` is not yet a field (the `model_copy` update adds an attribute Pydantic ignores / assertion on `r.selection_score` raises `AttributeError`).

- [ ] **Step 3: Add the field to `TrainingResult`**

In `src/mlops_agents/contracts/training.py`, the `TrainingResult` class currently ends:

```python
    champion_metrics: dict[str, float]
    forecast_chart_png: str | None = None  # base64 PNG; only set for forecasting runs
```

Add `selection_score` after it:

```python
    champion_metrics: dict[str, float]
    forecast_chart_png: str | None = None  # base64 PNG; only set for forecasting runs
    selection_score: float | None = None   # validation score the champion was selected on (metric_to_optimize units)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_contracts/test_training.py -k selection_score -v`
Expected: PASS

- [ ] **Step 5: Add the field to `TrainingStateUpdate` and map it**

In `src/mlops_agents/contracts/outputs.py`, the `TrainingStateUpdate` class lists fields ending with:

```python
    champion_candidate: dict | None = None
    experience_record_path: str | None = None
    forecast_chart_png: str | None = None
```

Add `selection_score`:

```python
    champion_candidate: dict | None = None
    experience_record_path: str | None = None
    forecast_chart_png: str | None = None
    selection_score: float | None = None
```

Then in `from_training_result`, the constructor call currently ends:

```python
            experience_record_path=result.experience_record_path,
            forecast_chart_png=result.forecast_chart_png,
        )
```

Add the mapping:

```python
            experience_record_path=result.experience_record_path,
            forecast_chart_png=result.forecast_chart_png,
            selection_score=result.selection_score,
        )
```

- [ ] **Step 6: Add the key to `AgentState`**

In `src/mlops_agents/state/agent_state.py`, the executor section currently ends:

```python
    experience_record_path: str | None     # JSON experience record serialised to disk
    forecast_chart_png: str | None         # base64 PNG chart; only set for forecasting runs
```

Add:

```python
    experience_record_path: str | None     # JSON experience record serialised to disk
    forecast_chart_png: str | None         # base64 PNG chart; only set for forecasting runs
    selection_score: float | None          # validation score the champion was selected on (forecasting)
```

- [ ] **Step 7: Run the contract + binding tests**

Run: `uv run pytest tests/test_contracts/ -q`
Expected: PASS — in particular `test_state_binding.py` confirms `selection_score` (newly in `AgentState`) is covered by `TrainingStateUpdate`.

- [ ] **Step 8: Commit**

```bash
git add src/mlops_agents/contracts/training.py src/mlops_agents/contracts/outputs.py src/mlops_agents/state/agent_state.py tests/test_contracts/test_training.py
git commit -m "feat(forecasting): add selection_score to TrainingResult, state update, and AgentState"
```

---

## Task 3: Implement `_build_test_exog` and `_forecast_champion_on_test` helpers

**Files:**
- Modify: `src/mlops_agents/training/executor.py` (add two module-level helpers near the other forecasting helpers, e.g. just above `_build_forecast_chart_png`)
- Test: `tests/test_training/test_forecast_champion_on_test.py` (create)

Context — existing helpers you will call (already in `executor.py`): `_is_statsforecast_model`, `_build_series_dict(df, dt_col, target, sid_cols, freq)`, `_resolve_exog_availability(columns, task_metadata) -> dict[col, "known_future"|"unknown_future"]`, `_fc_metrics(y_true, y_pred) -> {"rmse","mae",["mape"],"smape"}`. From `mlops_agents.training.exog_extender`: `extend_exog(history, horizon, strategy, freq) -> (pd.Series, fail|None)` and `align_val_exog_index(val_exog, series_dict, train_len, dt_col, freq)`. `ForecastingSettings` is already imported in `executor.py`.

- [ ] **Step 1: Write the failing test (no-leakage exog construction)**

Create `tests/test_training/test_forecast_champion_on_test.py`:

```python
import numpy as np
import pandas as pd
import pytest

from mlops_agents.contracts.training import ExogStrategySettings, ForecastingSettings, ValidationStrategy
from mlops_agents.training.executor import _build_series_dict, _build_test_exog


def _fs() -> ForecastingSettings:
    return ForecastingSettings(
        validation_strategy=ValidationStrategy(type="single_split", n_folds=1, horizon=3),
        exog_strategies=ExogStrategySettings(default_unknown_future="naive_carry"),
    )


def _task_meta() -> dict:
    return {
        "problem_type": "forecasting",
        "target_column": "y",
        "datetime_column": "ds",
        "series_id_columns": [],
        "frequency": "W",
        "forecast_horizon": 3,
        # _resolve_exog_availability reads `exogenous_columns` (list of {name, future_availability}).
        # 'holiday' is known-future (calendar); 'temp' is unknown-future.
        "exogenous_columns": [
            {"name": "temp", "future_availability": "unknown_future"},
            {"name": "holiday", "future_availability": "known_future"},
        ],
    }


def test_build_test_exog_extends_unknown_future_uses_actual_known_future():
    horizon = 3
    train = pd.DataFrame({
        "ds": pd.date_range("2023-01-02", periods=10, freq="W-MON"),
        "y": np.arange(10, dtype=float),
        "temp": np.arange(10, dtype=float),        # last value = 9.0
        "holiday": np.zeros(10, dtype=float),
    })
    test = pd.DataFrame({
        "ds": pd.date_range("2023-03-13", periods=horizon, freq="W-MON"),
        "y": [10.0, 11.0, 12.0],
        "temp": [100.0, 101.0, 102.0],             # very different from naive extension
        "holiday": [1.0, 0.0, 1.0],
    })
    series_dict = _build_series_dict(train, "ds", "y", [], "W")
    exog = _build_test_exog(train, test, _task_meta(), _fs(), horizon, "W", series_dict)

    # unknown-future temp must be the naive_carry extension (last train value 9.0), NOT the test actuals
    assert np.allclose(exog["temp"].to_numpy(), 9.0)
    assert not np.allclose(exog["temp"].to_numpy(), test["temp"].to_numpy())
    # known-future holiday uses the ACTUAL test values
    assert np.array_equal(exog["holiday"].to_numpy(), test["holiday"].to_numpy())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_training/test_forecast_champion_on_test.py -k build_test_exog -v`
Expected: FAIL with `ImportError` / `cannot import name '_build_test_exog'`.

- [ ] **Step 3: Implement `_build_test_exog`**

Add to `src/mlops_agents/training/executor.py` (above `_build_forecast_chart_png`):

```python
def _build_test_exog(
    train_pool: pd.DataFrame,
    test_df: pd.DataFrame,
    task_metadata: dict[str, Any],
    forecasting_settings: ForecastingSettings,
    horizon: int,
    freq: str | None,
    series_dict: dict[str, pd.Series],
) -> pd.DataFrame | None:
    """Build the test-horizon exog for a skforecast champion.

    Mirrors validation: known_future columns use the actual test values;
    unknown_future columns are extended from train history (no `drop` — all
    exog kept). No oracle peeking at unknown-future actuals.
    """
    dt_col = task_metadata["datetime_column"]
    availability = _resolve_exog_availability(list(train_pool.columns), task_metadata)
    strategies = forecasting_settings.exog_strategies
    future_values: dict[str, pd.Series] = {}
    for col, avail in availability.items():
        if col not in train_pool.columns:
            continue
        if avail == "known_future":
            future_values[col] = test_df[col].reset_index(drop=True)
        else:
            strat = strategies.per_column.get(col, strategies.default_unknown_future)
            preds_col, _ = extend_exog(train_pool[col], horizon, strat, freq)
            future_values[col] = preds_col.reset_index(drop=True)
    if not future_values:
        return None
    return align_val_exog_index(
        pd.DataFrame(future_values),
        series_dict,
        train_len=len(next(iter(series_dict.values()))),
        dt_col=dt_col,
        freq=freq,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_training/test_forecast_champion_on_test.py -k build_test_exog -v`
Expected: PASS

- [ ] **Step 5: Write the failing test (statsforecast happy path + graceful guard)**

Append to `tests/test_training/test_forecast_champion_on_test.py`:

```python
import pickle
from pathlib import Path
from mlops_agents.training.executor import _forecast_champion_on_test, _retrain_forecasting
from mlops_agents.models.loader import get_model


def test_forecast_champion_on_test_statsforecast(tmp_path):
    horizon = 4
    n = 40
    ds = pd.date_range("2023-01-02", periods=n, freq="W-MON")
    y = np.linspace(100, 200, n)  # clear ramp
    df = pd.DataFrame({"ds": ds, "y": y})
    train_pool = df.iloc[:-horizon].reset_index(drop=True)
    test_df = df.iloc[-horizon:].reset_index(drop=True)
    test_path = tmp_path / "test.csv"
    test_df.to_csv(test_path, index=False)

    task_meta = {
        "problem_type": "forecasting", "target_column": "y", "datetime_column": "ds",
        "series_id_columns": [], "frequency": "W", "forecast_horizon": horizon,
    }
    champion = {"model_key": "ets", "best_params": {"season_length": 1}, "best_score": 1.23}
    spec = get_model("ets")
    models_dir = tmp_path / "models"; models_dir.mkdir()
    champ_path = _retrain_forecasting(spec, champion, train_pool, task_meta, models_dir)

    fs = ForecastingSettings(
        validation_strategy=ValidationStrategy(type="single_split", n_folds=1, horizon=horizon),
        exog_strategies=ExogStrategySettings(),
    )
    metrics, preview = _forecast_champion_on_test(
        champion, champ_path, train_pool, test_path, task_meta, fs, "rmse",
    )
    assert "rmse" in metrics and metrics["rmse"] >= 0.0
    assert len(preview) == horizon
    assert set(preview[0].keys()) == {"ds", "y_true", "y_pred"}
    # y_true in the preview matches the held-out test target
    assert np.allclose([p["y_true"] for p in preview], test_df["y"].to_numpy())
```

- [ ] **Step 6: Run test to verify it fails**

Run: `uv run pytest tests/test_training/test_forecast_champion_on_test.py -k statsforecast -v`
Expected: FAIL with `cannot import name '_forecast_champion_on_test'`.

- [ ] **Step 7: Implement `_forecast_champion_on_test`**

Add to `src/mlops_agents/training/executor.py` (directly below `_build_test_exog`):

```python
def _forecast_champion_on_test(
    champion: dict,
    champion_model_path: Path,
    train_pool: pd.DataFrame,
    test_path: Path,
    task_metadata: dict[str, Any],
    forecasting_settings: ForecastingSettings,
    metric: str,
) -> tuple[dict[str, float], list[dict]]:
    """Forecast the retrained champion across the held-out test horizon.

    Returns (test_metrics, test_preview) where test_preview is
    [{"ds": str, "y_true": float, "y_pred": float}, ...] for the chart.
    statsforecast -> predict(h); skforecast -> extend unknown-future exog from
    train history, use actual known-future values, then predict(steps, exog).
    """
    target = task_metadata["target_column"]
    dt_col = task_metadata["datetime_column"]
    sid_cols = task_metadata.get("series_id_columns") or []
    horizon = int(task_metadata["forecast_horizon"])
    freq = task_metadata.get("frequency")

    test_df = pd.read_csv(test_path)
    test_df[dt_col] = pd.to_datetime(test_df[dt_col])
    test_df = test_df.sort_values(dt_col).reset_index(drop=True)
    y_true = test_df[target].to_numpy(dtype=float)

    with champion_model_path.open("rb") as f:
        model = pickle.load(f)

    if _is_statsforecast_model(champion["model_key"]):
        fcst = model.predict(h=horizon).sort_values("ds").reset_index(drop=True)
        model_col = [c for c in fcst.columns if c not in ("unique_id", "ds")][0]
        y_pred = fcst[model_col].to_numpy(dtype=float)
        ds_vals = pd.to_datetime(fcst["ds"]).dt.strftime("%Y-%m-%d").tolist()
    else:
        pool = train_pool.copy()
        pool[dt_col] = pd.to_datetime(pool[dt_col])
        series_dict = _build_series_dict(pool, dt_col, target, sid_cols, freq)
        test_exog = _build_test_exog(
            pool, test_df, task_metadata, forecasting_settings, horizon, freq, series_dict
        )
        preds = model.predict(steps=horizon, exog=test_exog)
        preds = preds.reset_index().rename(columns={"index": "ds"})
        preds["ds"] = pd.to_datetime(preds["ds"])
        y_pred = preds["pred"].to_numpy(dtype=float)
        ds_vals = preds["ds"].dt.strftime("%Y-%m-%d").tolist()

    n = min(len(y_true), len(y_pred))
    if n == 0:
        raise ValueError("test forecast produced no overlapping points")
    y_true, y_pred, ds_vals = y_true[:n], y_pred[:n], ds_vals[:n]
    test_metrics = _fc_metrics(y_true, y_pred)
    test_preview = [
        {"ds": ds_vals[i], "y_true": float(y_true[i]), "y_pred": float(y_pred[i])}
        for i in range(n)
    ]
    return test_metrics, test_preview
```

- [ ] **Step 8: Run test to verify it passes**

Run: `uv run pytest tests/test_training/test_forecast_champion_on_test.py -k statsforecast -v`
Expected: PASS

- [ ] **Step 9: Verify `Path` and `pickle` are imported in executor.py**

`executor.py` already imports `pickle` (top of file) and `from pathlib import Path`. Confirm with:

Run: `uv run python -c "import mlops_agents.training.executor as e; assert hasattr(e, '_forecast_champion_on_test') and hasattr(e, '_build_test_exog'); print('ok')"`
Expected: prints `ok`

- [ ] **Step 10: Run the full helper test file**

Run: `uv run pytest tests/test_training/test_forecast_champion_on_test.py -q`
Expected: PASS (all tests)

- [ ] **Step 11: Commit**

```bash
git add src/mlops_agents/training/executor.py tests/test_training/test_forecast_champion_on_test.py
git commit -m "feat(forecasting): add honest test-set forecast helpers with leakage-safe exog"
```

---

## Task 4: Rewire the forecasting branch in `run_training_plan`

**Files:**
- Modify: `src/mlops_agents/training/executor.py` (`fit_score` cleanup; champion retrain block; champion-metrics branch; MLflow logging; `TrainingResult` construction)
- Test: `tests/test_training/test_executor_forecasting.py`

- [ ] **Step 1: Remove the `drop` branch in `fit_score`**

In `src/mlops_agents/training/executor.py`, in the skforecast section of `fit_score`, change:

```python
                strat = strategies.per_column.get(col, strategies.default_unknown_future)
                if strat == "drop":
                    continue
                cache_key = (col, strat, fold_id, "default")
```

to:

```python
                strat = strategies.per_column.get(col, strategies.default_unknown_future)
                cache_key = (col, strat, fold_id, "default")
```

- [ ] **Step 2: Remove the `_val_preview` capture (chart now uses test preview)**

In `fit_score`, delete the `_val_preview` initialization line near the top of `_run_candidate_forecasting`:

```python
    _val_preview: list[dict] = []  # last fold's y_true vs pred; mutated by fit_score
```

In the statsforecast branch, delete both capture blocks so it reads:

```python
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
```

In the skforecast branch, delete the capture block so it reads:

```python
            if joined.empty:
                score = _fc_metrics(
                    val_long["y_true"].values, preds["pred"].values[: len(val_long)],
                )[metric]
            else:
                score = _fc_metrics(joined["y_true"].values, joined["pred"].values)[metric]
            fold_scores.append(score)
```

And remove `"last_val_preview": list(_val_preview),` from the candidate result dict returned by `_run_candidate_forecasting`.

- [ ] **Step 3: Run candidate-forecasting unit tests to confirm no regression**

Run: `uv run pytest tests/test_training/test_executor_forecasting.py -q -m "not integration"`
Expected: PASS (these tests don't assert on `last_val_preview`).

- [ ] **Step 4: Rewire the champion retrain + metrics branch**

In `run_training_plan`, the forecasting retrain block currently reads:

```python
        else:
            champion_path = _retrain_forecasting(spec, champion, train_pool, task_metadata, models_dir)
            val_preview = champion.get("last_val_preview") or []
            dt_col = task_metadata["datetime_column"]
            forecast_chart_png = (
                _build_forecast_chart_png(train_pool, val_preview, dt_col, target_column)
                if val_preview else None
            )
```

Replace it with just the retrain (chart moves to the metrics branch):

```python
        else:
            champion_path = _retrain_forecasting(spec, champion, train_pool, task_metadata, models_dir)
```

Ensure `forecast_chart_png` and `selection_score` are initialized once before the `if plan.problem_type in (...)` block. The classification/regression retrain branch already sets `forecast_chart_png = None`; add `selection_score` initialization next to it. Find:

```python
        if plan.problem_type in ("classification", "regression"):
            champion_path = _retrain_tabular(spec, champion, train_pool, target_column, models_dir)
            forecast_chart_png: str | None = None
```

and change to:

```python
        selection_score: float | None = None
        if plan.problem_type in ("classification", "regression"):
            champion_path = _retrain_tabular(spec, champion, train_pool, target_column, models_dir)
            forecast_chart_png: str | None = None
```

- [ ] **Step 5: Replace the forecasting champion-metrics computation**

The champion-metrics block currently reads:

```python
        if plan.problem_type in ("classification", "regression"):
            _test_df = pd.read_csv(test_path)
            _X_test = _test_df.drop(columns=[target_column])
            _y_test = _test_df[target_column]
            if label_encoder is not None:
                _y_test = pd.Series(label_encoder.transform(_y_test), index=_y_test.index)
            with champion_path.open("rb") as _f:
                _eval_model = pickle.load(_f)
            if plan.problem_type == "classification":
                all_champion_metrics = _cls_metrics(_y_test, _eval_model.predict(_X_test))
            else:
                all_champion_metrics = _reg_metrics(_y_test, _eval_model.predict(_X_test))
        else:
            all_champion_metrics = {metric: champion["best_score"]}
        mlflow.log_metrics(all_champion_metrics)
        logger.info(f"[executor] champion metrics: {all_champion_metrics}")
```

Replace the `else` branch (forecasting) and the MLflow logging:

```python
        if plan.problem_type in ("classification", "regression"):
            _test_df = pd.read_csv(test_path)
            _X_test = _test_df.drop(columns=[target_column])
            _y_test = _test_df[target_column]
            if label_encoder is not None:
                _y_test = pd.Series(label_encoder.transform(_y_test), index=_y_test.index)
            with champion_path.open("rb") as _f:
                _eval_model = pickle.load(_f)
            if plan.problem_type == "classification":
                all_champion_metrics = _cls_metrics(_y_test, _eval_model.predict(_X_test))
            else:
                all_champion_metrics = _reg_metrics(_y_test, _eval_model.predict(_X_test))
        else:
            selection_score = float(champion["best_score"])
            try:
                all_champion_metrics, _test_preview = _forecast_champion_on_test(
                    champion, champion_path, train_pool, test_path,
                    task_metadata, fs, metric,
                )
                forecast_chart_png = _build_forecast_chart_png(
                    train_pool, _test_preview, task_metadata["datetime_column"], target_column
                )
            except Exception as exc:
                logger.warning(f"[executor] test forecast failed: {exc}")
                all_champion_metrics = {}
                forecast_chart_png = None
        mlflow.log_metrics(all_champion_metrics)
        if plan.problem_type == "forecasting" and selection_score is not None:
            mlflow.log_metric(f"selection_{metric}", selection_score)
        logger.info(f"[executor] champion metrics: {all_champion_metrics}")
```

(`fs` is the resolved `ForecastingSettings` already in scope; `test_path` is the path returned by `split_dataset`.)

- [ ] **Step 6: Pass `selection_score` into `TrainingResult`**

The `TrainingResult(...)` construction currently ends:

```python
        champion_metrics=all_champion_metrics,
        forecast_chart_png=forecast_chart_png,
    )
```

Change to:

```python
        champion_metrics=all_champion_metrics,
        forecast_chart_png=forecast_chart_png,
        selection_score=selection_score,
    )
```

- [ ] **Step 7: Write the failing test (test-eval semantics)**

Add to `tests/test_training/test_executor_forecasting.py`:

```python
def test_forecasting_reports_test_metrics_and_selection_score(air_passengers_csv, tmp_path, monkeypatch):
    monkeypatch.setattr("mlops_agents.training.executor.settings.experience_pool_dir", tmp_path / "pool")
    from mlops_agents.contracts.training import TrainingPlan, TrainingPlanCandidate, TrialBudget
    from mlops_agents.training.executor import run_training_plan

    plan = TrainingPlan(
        problem_type="forecasting",
        candidates=[TrainingPlanCandidate(priority=1, model_key="ets")],
        trial_budget=TrialBudget(total_trials=3, min_trials_per_candidate=3, max_trials_per_candidate=3),
    )
    result = run_training_plan(
        plan=plan,
        processed_dataset_path=air_passengers_csv,
        target_column="passengers",
        task_metadata={
            "problem_type": "forecasting", "target_column": "passengers",
            "datetime_column": "month", "series_id_columns": [],
            "frequency": "MS", "forecast_horizon": 12,
        },
        output_dir=tmp_path / "splits",
        mlflow_experiment="test-air-honest",
        random_state=42,
    )
    # champion_metrics are now the TEST metrics (full _fc_metrics dict), not a single validation value
    assert "rmse" in result.champion_metrics
    assert "smape" in result.champion_metrics
    # selection_score is the validation score, recorded separately
    assert result.selection_score is not None
    # chart built from the test forecast
    assert result.forecast_chart_png is not None
```

- [ ] **Step 8: Run test to verify it fails, then passes**

Run: `uv run pytest tests/test_training/test_executor_forecasting.py -k honest -v`
Expected: PASS once Steps 4-6 are in place. (If you run it before editing, it fails because `champion_metrics` had only `rmse` and `selection_score` did not exist.)

- [ ] **Step 9: Run the full forecasting executor suite**

Run: `uv run pytest tests/test_training/ -q -m "not integration"`
Expected: PASS. Update any existing forecasting test that asserted `champion_metrics == {"rmse": <validation>}` to expect the multi-key test metrics + `selection_score` instead.

- [ ] **Step 10: Commit**

```bash
git add src/mlops_agents/training/executor.py tests/test_training/test_executor_forecasting.py
git commit -m "feat(forecasting): report held-out test metrics + selection_score; log test as rmse, validation as selection_metric"
```

---

## Task 5: Forward `selection_score` in the SSE event

**Files:**
- Modify: `api/services/pipeline.py` (`training_complete` event)
- Test: covered by frontend + manual; add a lightweight assertion if an event-builder unit test exists.

- [ ] **Step 1: Add `selection_score` to the event data**

In `api/services/pipeline.py`, the `training_complete` event currently builds:

```python
                            "data": {
                                "training_run_id":    ex.get("training_run_id", ""),
                                "training_metrics":   ex.get("training_metrics", {}),
                                "champion_candidate": ex.get("champion_candidate", {}),
                                "trained_model_path": ex.get("trained_model_path", ""),
                                "forecast_chart_png": ex.get("forecast_chart_png"),
                            },
```

Change to:

```python
                            "data": {
                                "training_run_id":    ex.get("training_run_id", ""),
                                "training_metrics":   ex.get("training_metrics", {}),
                                "champion_candidate": ex.get("champion_candidate", {}),
                                "trained_model_path": ex.get("trained_model_path", ""),
                                "forecast_chart_png": ex.get("forecast_chart_png"),
                                "selection_score":    ex.get("selection_score"),
                            },
```

- [ ] **Step 2: Verify the module imports cleanly**

Run: `uv run python -c "import api.services.pipeline; print('ok')"`
Expected: prints `ok`

- [ ] **Step 3: Commit**

```bash
git add api/services/pipeline.py
git commit -m "feat(api): forward selection_score in training_complete event"
```

---

## Task 6: Frontend — test metric, selection line, chart label, unavailable case

**Files:**
- Modify: `frontend/components/pipeline/ResultsDashboard.tsx` (`TrainingCompletePanel` signature + body; `trainingData` type)

- [ ] **Step 1: Extend the `trainingData` type**

In `frontend/components/pipeline/ResultsDashboard.tsx`, the `trainingData` memo casts the event. Change:

```tsx
    return ev ? (ev.data as { training_run_id: string; training_metrics: Record<string, number>; champion_candidate: Record<string, unknown>; trained_model_path: string; forecast_chart_png?: string | null }) : null
```

to add `selection_score`:

```tsx
    return ev ? (ev.data as { training_run_id: string; training_metrics: Record<string, number>; champion_candidate: Record<string, unknown>; trained_model_path: string; forecast_chart_png?: string | null; selection_score?: number | null }) : null
```

- [ ] **Step 2: Extend the `TrainingCompletePanel` prop type and body**

Change the function signature:

```tsx
function TrainingCompletePanel({ data }: { data: { training_run_id: string; training_metrics: Record<string, number>; champion_candidate: Record<string, unknown>; trained_model_path: string; forecast_chart_png?: string | null } }) {
```

to:

```tsx
function TrainingCompletePanel({ data }: { data: { training_run_id: string; training_metrics: Record<string, number>; champion_candidate: Record<string, unknown>; trained_model_path: string; forecast_chart_png?: string | null; selection_score?: number | null } }) {
```

Then change the metrics region. Currently:

```tsx
      {metricEntries.length > 0 && (
        <div>
          <p className="mb-1.5 text-xs font-medium text-zinc-500">Champion metrics</p>
          <div className="grid grid-cols-2 gap-2">
            {metricEntries.map(([k, v]) => (
              <div key={k} className="rounded border border-zinc-200 bg-zinc-50 px-2 py-1.5 text-xs">
                <div className="text-zinc-500">{k}</div>
                <div className="font-mono text-zinc-800">{typeof v === 'number' ? v.toFixed(4) : String(v)}</div>
              </div>
            ))}
          </div>
        </div>
      )}
```

Replace with a test-metrics heading + secondary validation line + an unavailable case:

```tsx
      {metricEntries.length > 0 ? (
        <div>
          <p className="mb-1.5 text-xs font-medium text-zinc-500">Test metrics (held-out)</p>
          <div className="grid grid-cols-2 gap-2">
            {metricEntries.map(([k, v]) => (
              <div key={k} className="rounded border border-zinc-200 bg-zinc-50 px-2 py-1.5 text-xs">
                <div className="text-zinc-500">{k}</div>
                <div className="font-mono text-zinc-800">{typeof v === 'number' ? v.toFixed(4) : String(v)}</div>
              </div>
            ))}
          </div>
          {typeof data.selection_score === 'number' && (
            <p className="mt-1.5 text-[11px] text-zinc-400">
              Selected on validation: <span className="font-mono">{data.selection_score.toFixed(4)}</span>
            </p>
          )}
        </div>
      ) : (
        <div>
          <p className="text-xs text-amber-600">Test evaluation unavailable.</p>
          {typeof data.selection_score === 'number' && (
            <p className="mt-1 text-[11px] text-zinc-400">
              Selected on validation: <span className="font-mono">{data.selection_score.toFixed(4)}</span>
            </p>
          )}
        </div>
      )}
```

- [ ] **Step 3: Relabel the chart heading**

Change:

```tsx
          <p className="mb-1.5 text-xs font-medium text-zinc-500">Forecast vs actuals</p>
```

to:

```tsx
          <p className="mb-1.5 text-xs font-medium text-zinc-500">Test forecast vs actuals</p>
```

(The chart image legend itself — "Train (actual)/Test (actual)/Test (predicted)" — is produced by `_build_forecast_chart_png`; update that legend text in Step 4.)

- [ ] **Step 4: Update the chart legend labels in the backend chart builder**

In `src/mlops_agents/training/executor.py`, `_build_forecast_chart_png`, the legend labels currently say "Validation (actual)/(predicted)". Change:

```python
        ax.plot(val_ds, val_true, color="#6b7280", linewidth=1.5, label="Validation (actual)")
        ax.plot(val_ds, val_pred, color="#f97316", linewidth=1.5, linestyle="--", label="Validation (predicted)")
```

to:

```python
        ax.plot(val_ds, val_true, color="#6b7280", linewidth=1.5, label="Test (actual)")
        ax.plot(val_ds, val_pred, color="#f97316", linewidth=1.5, linestyle="--", label="Test (predicted)")
```

(The parameter names inside the function stay as-is; only the displayed labels change.)

- [ ] **Step 5: Type-check / build the frontend**

Run: `cd frontend && npm run build` (or `npx tsc --noEmit`)
Expected: no type errors related to `selection_score` / `TrainingCompletePanel`.

- [ ] **Step 6: Commit**

```bash
git add frontend/components/pipeline/ResultsDashboard.tsx src/mlops_agents/training/executor.py
git commit -m "feat(ui): show held-out test metrics + validation selection line; relabel test forecast chart"
```

---

## Task 7: Rollout migration (one-time, manual ops)

**Files:** none (operational step performed at deploy time).

- [ ] **Step 1: Clear the historical MLflow runs so the gate baseline is test-based**

Because the promotion gate (`_fetch_current_champion`) ranks runs by the logged `rmse`, old runs that logged *validation* rmse must be cleared so new *test*-based runs aren't compared against them. Once, at rollout, delete the runs in the `mlops-agents` experiment (or rename the experiment to a fresh name in `settings.mlflow_experiment_name`).

Option A — delete via MLflow UI: open the MLflow server (http://localhost:5000), select the `mlops-agents` experiment, delete its runs.

Option B — delete the experiment programmatically (run against the same tracking URI the app uses):

```bash
uv run python -c "from mlflow.tracking import MlflowClient; c=MlflowClient('http://localhost:5000'); e=c.get_experiment_by_name('mlops-agents'); c.delete_experiment(e.experiment_id) if e else print('no experiment')"
```

Expected: the next forecasting run finds no champion → `evaluation_passed = True` (auto-passes), and subsequent runs compare test-vs-test.

Note: any Model-Registry versions pointing at deleted runs will dangle — harmless for this dev/thesis project.

- [ ] **Step 2: Smoke-test end to end**

Re-run the 3-year `grid_demand` forecasting pipeline through the UI. Confirm:
- Model tab shows **Test metrics (held-out)** with rmse/mae/smape and a "Selected on validation" line.
- The chart legend reads "Test (actual)/Test (predicted)".
- The experience-pool JSON for the run is written to the host (`experience_pool/grid_demand_forecast_forecasting_*.json`).

---

## Self-Review

**1. Spec coverage**

- Honest test eval (helper, approach B) → Task 3 + Task 4.
- Display: test prominent + secondary validation line + test chart → Task 6 (+ chart legend Task 6 Step 4).
- Forecasting-only scope → Task 4 only touches the forecasting branch; cls/reg untouched.
- Exog: mirror validation, no oracle → Task 3 `_build_test_exog` + its no-leakage test.
- Eliminate `drop` (6 locations) → Task 1 (contracts, planner validation, planner prompt) + Task 4 Step 1 (`fit_score` branch). `_retrain_forecasting` already uses all exog → now consistent.
- Promotion gate on test; log test as `rmse`, validation as `selection_{metric}` → Task 4 Step 5.
- `selection_score` through contracts/state/event/UI → Tasks 2, 5, 6.
- train + validation training for test → uses `_retrain_forecasting(train_pool)` (documented in spec; no code change).
- Failure handling (thin guard, no relabel) → Task 4 Step 5 try/except + Task 6 unavailable case.
- Migration (clear MLflow experiment) → Task 7.
- Deployment refit-on-all-data → explicitly out of scope (spec known-limitations); no task.
- Extension quality (ets vs naive_carry) → out of scope (SP2); default `naive_carry` retained.

**2. Placeholder scan:** none — every code step shows full old/new content and exact commands.

**3. Type consistency:** `selection_score: float | None` is consistent across `TrainingResult`, `TrainingStateUpdate`, `AgentState`, the event dict, and the frontend (`selection_score?: number | null`). `_forecast_champion_on_test` and `_build_test_exog` signatures match their call sites in Task 4. `champion_metrics` is a `dict[str, float]` everywhere (now multi-key for forecasting). MLflow key is `selection_{metric}` consistently.
