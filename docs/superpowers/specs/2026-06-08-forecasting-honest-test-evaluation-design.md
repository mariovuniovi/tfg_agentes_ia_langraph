# SP1 — Honest Test-Set Evaluation for Forecasting

**Date:** 2026-06-08
**Status:** Design approved + grilled; pending final spec review
**Scope:** Sub-project 1 of the forecasting-correctness effort. SP2 (smarter planner/profiler *decisions*: which exog extension strategy, short-history exog-model eligibility, recent-window trend detection, small-data model-selection guardrails) is a separate spec.

## Problem

For forecasting runs, the metric and chart shown in the Model tab are the **validation** result, not the held-out **test** result — but they are labeled as if they were the model's forecast.

In `run_training_plan` ([executor.py](../../../src/mlops_agents/training/executor.py)) the forecasting branch does:

```python
else:  # forecasting
    all_champion_metrics = {metric: champion["best_score"]}   # validation score, not test
```

Classification and regression already score the champion on the held-out test set ([executor.py:937-948](../../../src/mlops_agents/training/executor.py#L937)); forecasting skips this. Consequences observed on the `grid_demand` run:

- Reported RMSE **107.72** is the validation window (last 8 points of the train pool, a flat regime). The true test window (final 8 weeks, a steep climb) is never forecast, never scored, never charted.
- `forecast_chart_png` is built from the **validation** fold (`_val_preview`), so even the chart shows the validation fit, not the test forecast.
- A flat `ets` forecast looks acceptable (107 ≈ 4%) while its real test RMSE would be ~500–600.

The metric flatters the model because it measures the easy region.

A second, coupled defect surfaced during the design grill: exog handling is **inconsistent** across paths. `fit_score` (validation) skips `drop` columns ([executor.py:592-594](../../../src/mlops_agents/training/executor.py#L592)); `_retrain_forecasting` ([executor.py:755-758](../../../src/mlops_agents/training/executor.py#L755)) refits the champion with **all** exog, ignoring `drop`. So the validated model and the deployed/tested model can use different feature sets — which would make any "test" number describe a different model than validation selected. An honest test requires consistent exog handling, so `drop` removal is folded into SP1 as a prerequisite.

## Goal

Make the forecasting Model tab show the **honest held-out test performance** of the champion, surface the **validation/selection score** separately, and make exog handling **consistent** (no `drop`) so validation, retrain, and test all describe the same model.

Champion *selection* stays on validation (standard ML practice — selecting on the test set is leakage). SP1 changes *evaluation, reporting, and exog consistency*, not selection.

## In scope

1. Honest held-out **test** evaluation for forecasting (new helper, approach B).
2. Model tab shows **test** metric + **test** forecast chart; **selection (validation)** score shown as a secondary line.
3. **Eliminate the `drop` exog strategy** entirely (validation, retrain, test all use all exog; unknown-future extended) so the three paths are consistent.
4. **Promotion gate compares on test**: executor logs **test** metrics as `rmse`; validation logged separately as `selection_{metric}` (non-ranking). One-time MLflow experiment cleanup so the gate doesn't compare test-vs-validation across the transition.

## Non-goals (deferred to SP2 unless noted)

- **Extension *quality*** — choosing `ets`/`auto_arima` over the `naive_carry` default for seasonal continuous exog. SP1 removes `drop` and always extends; *which* non-drop strategy is SP2.
- Short-history exog-model eligibility rules; recent-window trend detection; small-data model-selection guardrails.
- Classification/regression evaluation (already correct) and their metric display.
- **Deployment refit-on-all-data** — the deployed artifact is the train-pool-retrained champion (never sees the test weeks). For forecasting you'd normally refit on all data before deploying. This is a *deployment-node* concern, out of SP1 scope; **noted as future work**.

## Key decisions (from brainstorming + grill)

1. **Display (option 1):** Model tab shows the **test** metric prominently + a small secondary line "Selected on validation {metric}: X". Chart shows the **test** forecast.
2. **Scope:** forecasting-only for the test-eval and dual display. Classification/regression unchanged.
3. **Exog in validation/retrain/test:** `drop` is removed. `known_future` columns use actual values (calendar — legitimately known ahead); `unknown_future` columns are **always extended** from train history via the configured non-drop strategy (default `naive_carry`); actual future values of unknown-future exog are never used (no oracle). Univariate statistical models (`naive`/`seasonal_naive`/`ets`/`auto_arima`) ignore exog automatically — their statsforecast factories don't accept exog.
4. **Test set = held-out tail; training = train + validation.** `split_dataset` holds out the last `horizon` rows as test; the champion is retrained on the entire train pool (= train + validation, since validation is CV folds within the train pool) before the test forecast. Already the case; documented to lock it.
5. **Promotion gate on test:** log test metrics as `rmse` (the metric `_fetch_current_champion` ranks on); log the validation/selection score as `selection_{metric}` (not ranked). One-time clear of the `mlops-agents` MLflow experiment runs at rollout so the gate baseline is test-based. First post-cleanup run has no champion → `evaluation_passed = True` (expected).
6. **Failure handling:** thin `try/except` around the test forecast. It reuses the same machinery validation already exercised, so failure is not expected; the guard only prevents a reporting step from crashing an otherwise-successful run (model + MLflow artifacts already saved). On failure: metric unavailable + reason, chart omitted, validation line still shown, **validation never relabeled as test**.
7. **Structure (approach B):** extract a focused, independently unit-testable helper; the Optuna validation loop is otherwise untouched.

## Architecture & data flow

```
run_training_plan (executor)
  ├─ retrain champion on train_pool (= train + validation)   (_retrain_forecasting → champion .pkl)
  ├─ selection_score = champion["best_score"]                # validation criterion (metric_to_optimize units)
  ├─ test_metrics, test_preview = _forecast_champion_on_test(...)   # NEW honest test
  ├─ mlflow.log_metrics(test_metrics)                        # test = primary `rmse` (gate ranks on this)
  ├─ mlflow.log_metric(f"selection_{metric}", selection_score)      # non-ranking traceability
  ├─ forecast_chart_png = _build_forecast_chart_png(train_pool, test_preview, ...)   # test, not val
  └─ TrainingResult(champion_metrics=test_metrics, selection_score=selection_score, forecast_chart_png=...)
        ↓
TrainingStateUpdate.from_training_result → AgentState → api/services/pipeline.py training_complete event
        ↓
promotion gate: candidate = training_metrics (test) vs _fetch_current_champion (now test-based)
        ↓
frontend ResultsDashboard TrainingCompletePanel
  - champion metrics grid = TEST metrics
  - secondary line        = "Selected on validation {metric}: {selection_score}"
  - chart                 = train + test-actual + test-predicted
```

## Components

### 1. `_forecast_champion_on_test` (new helper in executor.py)

```python
def _forecast_champion_on_test(
    champion: dict,
    champion_model_path: Path,
    train_pool: pd.DataFrame,
    test_path: Path,
    task_metadata: dict,
    forecasting_settings: ForecastingSettings,
    metric: str,
) -> tuple[dict[str, float], list[dict]]:
    """Forecast the retrained champion across the held-out test horizon.

    Returns (test_metrics, test_preview). test_preview is
    [{"ds": str, "y_true": float, "y_pred": float}, ...] for the chart.
    Exog handling mirrors validation (all exog; known_future = actual,
    unknown_future = extend_exog from train history; no oracle peeking).
    """
```

Behaviour:
- Reload the champion pickle (same pattern as the tabular test-eval at [executor.py:943](../../../src/mlops_agents/training/executor.py#L943)).
- **statsforecast champion** (`_is_statsforecast_model` true): `preds = sf.predict(h=horizon)`; align to test dates; no exog.
- **skforecast champion:** recompute `used_cols` via `_resolve_exog_availability` (identical to `_retrain_forecasting`, so it matches what the model was fit with — no `drop` branch now); `known_future` → actual test values from `test_path`; `unknown_future` → `extend_exog(train_pool[col], horizon, strategy, freq)`; align via `align_val_exog_index`; `forecaster.predict(steps=horizon, exog=test_exog)`.
- Load test target from `test_path`, align on the datetime column, compute `_fc_metrics(y_true, y_pred)` (rmse/mae/mape/smape).
- Build `test_preview` from the aligned (ds, y_true, y_pred).
- Reuses: `_is_statsforecast_model`, `_to_sf_format`, `_build_series_dict`, `_resolve_exog_availability`, `extend_exog`, `align_val_exog_index`, `_fc_metrics`.

### 2. Remove the `drop` exog strategy (consistency prerequisite)

- [contracts/training.py:18-19](../../../src/mlops_agents/contracts/training.py#L18): drop `"drop"` from `ExogStrategy` and `UnknownFutureStrategy` Literals. `default_unknown_future` stays `"naive_carry"`.
- [executor.py:603-604](../../../src/mlops_agents/training/executor.py#L603): delete the `if strat == "drop": continue` branch in `fit_score` — all unknown-future exog are always extended. (`_retrain_forecasting` already uses all exog → now consistent with validation.)
- [planning/validation.py:24](../../../src/mlops_agents/planning/validation.py#L24): remove `"drop"` from `ALLOWED_EXOG_STRATEGIES`; reword the line-113 message that mentions the "drop loophole" (the known_future check stays).
- [prompts/planner.yaml:48](../../../src/mlops_agents/prompts/planner.yaml#L48): remove `drop` from the strategy options shown to the planner.

### 3. Contracts

- [`TrainingResult`](../../../src/mlops_agents/contracts/training.py): add `selection_score: float | None = None`. Forecasting `champion_metrics` now holds **test** metrics; `selection_score` holds the validation score (in `metric_to_optimize` units). `forecast_chart_png` already exists.
- [`TrainingStateUpdate`](../../../src/mlops_agents/contracts/outputs.py): add `selection_score: float | None = None`; map it in `from_training_result`.
- [`AgentState`](../../../src/mlops_agents/state/agent_state.py): add `selection_score: float | None` in the executor section (kept in sync by the existing binding test).

### 4. Executor wiring (forecasting branch only) + MLflow

Replace the forecasting branch of the champion-metrics computation:
- `selection_score = champion["best_score"]`
- `test_metrics, test_preview = _forecast_champion_on_test(...)` (inside the thin guard)
- `all_champion_metrics = test_metrics`; build chart from `test_preview`
- `mlflow.log_metrics(test_metrics)` (test = primary `rmse`), `mlflow.log_metric(f"selection_{metric}", selection_score)` (non-ranking)

Classification/regression branches unchanged; their `selection_score` stays `None` (not surfaced).

### 5. Event

[api/services/pipeline.py](../../../api/services/pipeline.py) `training_complete` event `data`: add `selection_score: ex.get("selection_score")`. (`forecast_chart_png` already forwarded.)

### 6. Frontend (forecasting display)

`TrainingCompletePanel` in [ResultsDashboard.tsx](../../../frontend/components/pipeline/ResultsDashboard.tsx):
- Champion-metrics grid fed the **test** metrics (same `training_metrics` field, now carrying test values; now 4 keys: rmse/mae/mape/smape).
- New muted secondary line, rendered only when `selection_score` is present: `Selected on validation {metric}: {selection_score}`.
- Chart legend relabeled "Train (actual)" / "Test (actual)" / "Test (predicted)" (no separate validation segment).
- If `training_metrics` is empty (guard tripped): show "Test evaluation unavailable" + keep the validation line + omit the chart.
- `trainingData` type + `TrainingCompletePanel` props gain `selection_score?: number | null`.

## Error handling

`_forecast_champion_on_test` wrapped in a single `try/except` in `run_training_plan`:
- Success: `champion_metrics = test_metrics`, `forecast_chart_png` from `test_preview`.
- Failure: log warning; `champion_metrics = {}`, `forecast_chart_png = None`; `selection_score` still set; `mlflow.log_metric(selection_{metric})` still logged. Run completes; artifacts preserved; validation never relabeled as test.

## Migration (one-time, manual ops at rollout)

Clear the historical runs in the `mlops-agents` MLflow experiment (or rename to a fresh experiment) once, so `_fetch_current_champion` starts from a test-based baseline rather than comparing new test metrics against old validation metrics. Caveat: any Model-Registry versions pointing at deleted runs would dangle — harmless for this dev/thesis project (not in production).

## Testing

Unit tests for `_forecast_champion_on_test` (the point of approach B):
- **statsforecast:** synthetic series with a known ramp in the test window; assert returned test RMSE matches a hand-computed value within tolerance; assert `test_preview` dates equal the test dates.
- **skforecast / no oracle leakage:** construct a series where the unknown-future exog's naive extension differs sharply from its actual test values; assert the helper's prediction matches the *extended-exog* prediction, not the *actual-exog* one.
- **graceful guard:** empty/degenerate test set returns without raising.

Drop-removal:
- A plan/contract with `exog strategy = "drop"` is now rejected (Literal/validation no longer allow it).
- `fit_score` includes all unknown-future exog (no skip) — assert via a forecasting run that the model is fit with the exog columns present.

Integration / regression:
- For a forecasting run, `TrainingResult.champion_metrics` (test) and `selection_score` (validation) are distinct sources and differ when regimes differ.
- `selection_score` added to `AgentState` is covered by the existing contract binding test.
- Update existing forecasting executor tests for the new `champion_metrics` = test semantics and the removed `drop` option.

Unit tests must not make real LLM calls; forecasting executor tests use synthetic pandas frames and the default-params path (no real MLflow server required, per existing patterns).

## Success criteria

- A forecasting run's Model tab shows the champion's **test** RMSE and a **test** forecast chart, plus a secondary "Selected on validation {metric}" line.
- On the `grid_demand` 3-year dataset, the displayed test metric reflects true held-out forecast quality (no longer the flattering validation number), making the validation-vs-test gap visible.
- Validation, retrain, and test all use the **same** exog set (no `drop`); unknown-future exog is never sourced from actual future values.
- The promotion gate ranks on **test** `rmse`; the validation score is preserved as `selection_{metric}` in MLflow.
- A failure in the test-eval degrades visibly without crashing the run or relabeling validation as test.

## Known limitations / future work

- **Deployment refit-on-all-data:** deployed artifact is the train-pool model; refit on train+test before deployment is a separate deployment-node improvement.
- **Exog extension quality:** unknown-future exog is extended with `naive_carry` by default; SP2 will select `ets`/`auto_arima` for seasonal continuous exog.
