# SP2b′ — Deterministic Forecasting Policy (validation + exog out of the planner)

**Date:** 2026-06-09
**Status:** Design approved + grilled, pending final spec review
**Scope:** First sub-project of SP2. Sibling sub-projects (separate specs): **SP2a** — model eligibility (history-threshold / short-history rule excluding supervised+exog models) + recent-window trend detection; **SP2d** — dataset-aware promotion gate. SP2b′ replaces the original SP2b (validation strategy) and absorbs the architectural core of the original SP2c (exog extension).

## Problem

Two deterministic decisions are currently made by the **LLM planner**, and it gets them wrong:

1. **Validation strategy.** `planner.yaml` tells the LLM "if history is very_short/short → prefer single_split." On the 3-year `grid_demand` run (156 obs, bucketed `short`) the planner chose `single_split` — a single 8-point validation window. A flat `naive` forecast then won the champion bake-off on that one lucky window, despite seasonality being detected. Yet multi-fold backtesting is trivially affordable here (≈5 folds). It was skipped purely because of the coarse `short` label.

2. **Exog extension.** The planner also chooses how to project unknown-future exog (e.g. `avg_temp_c`) forward. It picked `naive_carry`, which flattens a strongly seasonal driver.

Neither is a reasoning task — both are capacity/profiling arithmetic. Per CLAUDE.md ("Deterministic first… agents only for interpreting failures, reasoning about strategy"), the LLM should not be choosing fold counts or exog extension. Model *selection* is the genuine reasoning task and stays with the planner.

## Goal

Move validation-strategy and exog-extension selection into **deterministic training-layer policies that resolve before the planner runs**. The planner receives the resolved settings as fixed context, selects **models only**, and may explain the policy decisions in its rationale. The same policy functions also back the executor's fallback path, so behaviour is identical for planner-driven, benchmark, and direct-executor runs.

## Key decisions (from brainstorming + grill)

1. **Resolve-before-planner ordering.** In `planner_node`'s pre-agent setup, compute `validation_strategy` (ValidationPolicy) and `exog_strategies` (ExogPolicy) deterministically, inject a readable summary into the planner prompt, and — after the agent returns — inject the resolved `ForecastingSettings` into the plan **before** the validation checks run. The LLM emits candidates + rationale only.
2. **Capacity-driven validation fold count** (not bucket-driven), with a hard-error floor.
3. **Per-column profiled exog extension** reusing existing machinery: `_detect_per_series` for the decision, `extend_exog` (which already falls back to `naive_carry` on fit failure) for execution. `ets` only (captures trend + seasonality); `auto_arima` left available but unused. Non-numeric columns are guarded → `naive_carry` (no crash).
4. **Shared policy functions for both call sites.** `resolve_validation_strategy` and `resolve_exog_strategies` are training-layer helpers called by both `planner_node` (pre-agent) and the `run_training_plan` fallback (when a plan arrives with `forecasting_settings=None`, e.g. benchmark runner / direct-executor tests). One behaviour everywhere.
5. **Remove the now-obsolete static guidance:** the 6 `ml_rules.yaml` validation/exog rules and the `planner.yaml` validation/exog instructions. The planner-side forecasting-settings validation (`_check_plan_integrity` step 5) **stays** as a cheap guard — it now validates the injected policy settings.
6. **Experience pool unchanged** — still records `validation_strategy_json` / `exog_strategies_json` as descriptive facts (no longer planner decision inputs).

## Non-goals (other SP2 sub-projects / future)
- Model eligibility / history threshold / supervised-model admission (**SP2a**).
- Recent-window trend detection (**SP2a**).
- Dataset-aware promotion gate (**SP2d**).
- Learning validation/exog strategy from the experience pool (meta-learning; out of scope).

## Architecture & data flow

```
data_validator → dataset_approval
  → planner_node (pre-agent setup):
       df       = read_csv(processed_dataset_path)          # planner_node currently only profiles; add this load
       profile  = build_dataset_profile(path, task_meta)
       n_obs    = len(df)                                    # exact under V1 single-series constraint
       vstrat   = resolve_validation_strategy(profile, task_meta, n_obs)   # capacity-driven, hard-error floor
       exog     = resolve_exog_strategies(df, task_meta, freq)             # per-column: ets if seasonal/trend else naive_carry
       fs       = ForecastingSettings(validation_strategy=vstrat, exog_strategies=exog)
       inject readable summary of fs into the planner prompt context
       → LLM agent: selects MODELS ONLY (+ rationale, may explain fs)
       → output.plan = output.plan.model_copy(update={"forecasting_settings": fs})   # inject BEFORE validation
       → _check_plan_integrity(output, ...)                 # now validates the policy settings (passes by construction)
  → executor run_training_plan:
       if plan.forecasting_settings is None:  # benchmark / direct-executor path
           fs = ForecastingSettings(resolve_validation_strategy(profile, task_meta, len(df)),
                                    resolve_exog_strategies(df, task_meta, freq))
       validate_forecasting_plan(...)  # still guards capacity/horizon/exog downstream
```

## Components

### 1. `resolve_validation_strategy` (extend `src/mlops_agents/training/validation_policy.py`)

Capacity-driven replacement for the bucket logic in the current `select_validation_strategy`:

```python
_MAX_FOLDS = 5

def resolve_validation_strategy(
    profile: DatasetProfile, task_metadata: dict, n_obs: int
) -> ValidationStrategy:
    horizon = int(task_metadata["forecast_horizon"])
    drift = task_metadata.get("expected_drift", "low")
    train_pool_len = n_obs - horizon                      # deterministic split: test = last horizon
    min_train = max(_HORIZON_MULTIPLIER * horizon, _MIN_TRAIN_ROWS)   # = max(3*h, 30)
    k_max = (train_pool_len - min_train) // horizon
    if k_max < 1:
        # need train_pool_len >= min_train + horizon, and train_pool_len = n_obs - horizon,
        # so n_obs >= min_train + 2*horizon (one horizon for the test split, one for the val window)
        raise ValueError(
            f"Not enough history for a single validation split: need >= "
            f"{min_train + 2 * horizon} observations for horizon {horizon}, have {n_obs}."
        )
    k = min(k_max, _MAX_FOLDS)
    if k == 1:
        return ValidationStrategy(type="single_split", n_folds=1, horizon=horizon)
    vtype = "rolling_window" if drift == "high" else "expanding_window"
    return ValidationStrategy(type=vtype, n_folds=k, horizon=horizon, step_size=horizon)
```

This **replaces** the bucket-based `select_validation_strategy`. Called from both `planner_node` and the executor fallback. `resolve_rolling_window_size` and `validate_forecasting_plan` stay — the executor still resolves rolling `window_size` and validates capacity, and the new policy + that guard agree by construction (`k*horizon + min_train <= train_pool_len`). `iter_folds` already implements `single_split` / `expanding_window` / `rolling_window`, so multi-fold needs no executor change.

### 2. `resolve_exog_strategies` (new `src/mlops_agents/training/exog_policy.py`)

```python
def resolve_exog_strategies(
    df: pd.DataFrame, task_metadata: dict, freq: str | None
) -> ExogStrategySettings:
    """Per-column deterministic extension strategy.

    known_future   -> handled as actual (not in per_column unknown-future map)
    unknown_future -> ets if the column itself is seasonal/trending, else naive_carry
    non-numeric    -> naive_carry (cannot profile; carry last value)
    """
    declared = task_metadata.get("exogenous_columns") or []
    per_column: dict[str, str] = {}
    for entry in declared:
        col, avail = entry["name"], entry["future_availability"]
        if avail != "unknown_future" or col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            per_column[col] = "naive_carry"            # guard: user-declared exog dtype is not guaranteed numeric
            continue
        seasonal, trend, _ = _detect_per_series(df[col].astype(float), freq)
        per_column[col] = "ets" if (seasonal or trend) else "naive_carry"
    return ExogStrategySettings(per_column=per_column, default_unknown_future="naive_carry")
```

Reuses `_detect_per_series` (from `profiler.py`) and relies on `extend_exog`'s existing `ets`→`naive_carry` fallback, so it is never worse than `naive_carry`. The non-numeric guard prevents a `astype(float)` crash on a user schema that declares a categorical `unknown_future` exog (the dtype comes from `schema_data["exogenous_columns"]`, not guaranteed numeric). If `_detect_per_series` is module-private, expose it via a thin public wrapper rather than duplicating logic.

### 3. `planner_node` wiring (`src/mlops_agents/planning/node.py`)

- `planner_node` currently only passes the path to `build_dataset_profile`; **add a `read_csv(processed_dataset_path)`** so both policies have the dataframe and `n_obs`.
- For forecasting problem types, call both policies to build `fs: ForecastingSettings`.
- Add a compact, human-readable summary of `fs` to the planner prompt input (e.g. via `format_planner_inputs`) so the LLM's model rationale is coherent with the resolved policy and it can explain it.
- After the agent returns `output`, **first** inject `output.plan = output.plan.model_copy(update={"forecasting_settings": fs})`, **then** run the existing validation checks. This removes the "missing forecasting_settings" failure mode (the LLM no longer emits it) and makes `_check_plan_integrity` step 5 validate the policy settings.

### 4. Remove obsolete planner-facing guidance

- `src/mlops_agents/knowledge/ml_rules.yaml`: delete the 6 rules under "exog & validation strategy guidance" — `forecasting_short_history_single_split`, `forecasting_medium_long_expanding_window`, `forecasting_high_drift_rolling_window`, `exog_calendar_known_future`, `exog_unknown_default_naive_carry`, `exog_slow_macro_auto_arima`.
- `src/mlops_agents/prompts/planner.yaml`: remove the validation-strategy and exog-strategy instruction lines (the LLM no longer chooses these).
- `src/mlops_agents/planning/validation.py`: **keep** `_check_plan_integrity` step 5 unchanged — it now guards the injected policy settings (passes by construction; the `known_future` sub-check, which reads `known_future_columns`, becomes inert but harmless).

### 5. Contracts

`ForecastingSettings` / `ValidationStrategy` / `ExogStrategySettings` unchanged — they remain the carrier, now populated by the policies. No change to the experience-record schema.

## Testing

- **`resolve_validation_strategy`** (`tests/test_training/test_validation_policy.py`), all at horizon 8 (min_train=30, single_split needs n_obs ≥ min_train+2*horizon = 46): 156 obs → `expanding_window`, n_folds=5 (capped); 60 obs → `expanding_window`, n_folds=2; 46 obs → `single_split`; 20 obs → raises with the actionable message ("need >= 46 observations for horizon 8, have 20"); `expected_drift="high"` with ample history → `rolling_window`.
- **`resolve_exog_strategies`** (new `tests/test_training/test_exog_policy.py`): a synthetic seasonal unknown-future column → `ets`; a flat/noise unknown-future column → `naive_carry`; a **non-numeric** unknown-future column → `naive_carry` (no crash); a `known_future` column → absent from `per_column`; no exog declared → empty `per_column`.
- **executor fallback** (extend `tests/test_training/test_executor_forecasting.py`): a `TrainingPlan` with `forecasting_settings=None` on ample data runs and ends up with a multi-fold `validation_strategy` (proving the fallback routes through `resolve_validation_strategy`, not a hardcoded single_split).
- **`planner_node`** (extend `tests/test_agents/test_planner_node.py`): with the LLM agent mocked to return a plan whose `forecasting_settings` is `None` *or* a deliberately wrong strategy, assert the node's resulting `training_plan.forecasting_settings` equals the policy-resolved settings — proving the LLM cannot override, and that injection-before-validation prevents a "missing forecasting_settings" failure.
- Update/remove existing planner tests that asserted LLM-chosen validation/exog, and any rule-matching test referencing the 6 deleted rules.
- Unit tests must not make real LLM calls (mock the agent); policy tests use synthetic pandas frames.

## Success criteria

- For the 3-year `grid_demand` dataset, the resolved `validation_strategy` is multi-fold (≈5-fold `expanding_window`), so champion selection is backtest-robust and a flat `naive` no longer wins on a single lucky window.
- A strongly seasonal unknown-future exog (`avg_temp_c`) resolves to `ets` extension, not `naive_carry`; a non-numeric exog resolves to `naive_carry` without crashing.
- The LLM planner produces candidates + rationale only; the plan's `forecasting_settings` is always the policy output, verified by test, for both planner-driven and benchmark/direct-executor runs.
- Genuinely tiny data (can't afford one clean split) raises a clear, actionable error rather than producing a fragile model.
- The 6 validation/exog rules and the planner-prompt guidance are gone; `mypy`/tests stay green.
