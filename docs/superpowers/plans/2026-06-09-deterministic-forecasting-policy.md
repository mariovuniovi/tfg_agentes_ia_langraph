# Deterministic Forecasting Policy (SP2b′) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Project commit policy:** CLAUDE.md says "Never commit changes" and "Never add Claude as co-author." Commit steps are included for structure; follow the user's policy — never add a `Co-Authored-By: Claude` trailer, and only commit if the user authorized it for this execution.

**Goal:** Move forecasting validation-strategy and exog-extension selection out of the LLM planner into deterministic training-layer policies that resolve before the planner runs, so the LLM selects models only.

**Architecture:** Two new pure functions — `resolve_validation_strategy(task_metadata, n_obs)` (capacity-driven fold count, hard-error floor) and `resolve_exog_strategies(df, task_metadata, freq)` (per-column: `ets` if seasonal/trending else `naive_carry`, numeric-guarded). Both are called by `planner_node` (pre-agent; result injected into the plan before validation) and by the `run_training_plan` fallback, so behaviour is identical everywhere. The 6 validation/exog rules in `ml_rules.yaml` and the validation/exog lines in `planner.yaml` are removed; model-eligibility guidance stays.

**Tech Stack:** Python 3.12, UV, pandas, statsforecast/skforecast, Pydantic, pytest.

**Spec:** `docs/superpowers/specs/2026-06-09-deterministic-forecasting-policy-design.md`

---

## File Structure

| File | Change |
|------|--------|
| `src/mlops_agents/training/validation_policy.py` | add `resolve_validation_strategy`; (Task 3) remove `select_validation_strategy` |
| `src/mlops_agents/training/profiler.py` | add public `detect_series_structure` wrapper around `_detect_per_series` |
| `src/mlops_agents/training/exog_policy.py` | **new** — `resolve_exog_strategies` |
| `src/mlops_agents/training/executor.py` | fallback uses the two policies; swap imports |
| `src/mlops_agents/planning/node.py` | load df, resolve policies, inject into prompt + plan before validation |
| `src/mlops_agents/planning/prompts.py` | `format_planner_inputs` gains an optional policy-summary arg |
| `src/mlops_agents/knowledge/ml_rules.yaml` | delete the 6 validation/exog rules |
| `src/mlops_agents/prompts/planner.yaml` | drop validation/exog guidance, keep model-eligibility guidance |
| `tests/test_training/test_validation_policy.py` | add `resolve_validation_strategy` tests; (Task 3) drop `select_validation_strategy` tests |
| `tests/test_training/test_exog_policy.py` | **new** |
| `tests/test_training/test_executor_forecasting.py` | fallback-routes-to-multifold test |
| `tests/test_planning/test_node.py` | policy-injection test |

---

## Task 1: `resolve_validation_strategy` (capacity-driven)

**Files:**
- Modify: `src/mlops_agents/training/validation_policy.py`
- Test: `tests/test_training/test_validation_policy.py`

Context: `validation_policy.py` already defines `_HORIZON_MULTIPLIER = 3`, `_MIN_TRAIN_ROWS = 30`, and imports `from typing import Any` and `ValidationStrategy`. Add the new function; do NOT remove `select_validation_strategy` yet (Task 3 does that, after the executor is switched over).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_training/test_validation_policy.py`:

```python
from mlops_agents.training.validation_policy import resolve_validation_strategy


def _vmeta(horizon=8, drift="low"):
    return {"forecast_horizon": horizon, "expected_drift": drift}


def test_resolve_ample_history_caps_at_5_folds():
    vs = resolve_validation_strategy(_vmeta(), n_obs=156)
    assert vs.type == "expanding_window"
    assert vs.n_folds == 5
    assert vs.horizon == 8


def test_resolve_moderate_history_two_folds():
    vs = resolve_validation_strategy(_vmeta(), n_obs=60)
    assert vs.type == "expanding_window"
    assert vs.n_folds == 2


def test_resolve_floor_is_single_split():
    vs = resolve_validation_strategy(_vmeta(), n_obs=46)
    assert vs.type == "single_split"
    assert vs.n_folds == 1


def test_resolve_too_small_raises():
    import pytest
    with pytest.raises(ValueError, match="need >= 46 observations for horizon 8, have 20"):
        resolve_validation_strategy(_vmeta(), n_obs=20)


def test_resolve_high_drift_uses_rolling_window():
    vs = resolve_validation_strategy(_vmeta(drift="high"), n_obs=156)
    assert vs.type == "rolling_window"
    assert vs.n_folds == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_training/test_validation_policy.py -k resolve -v`
Expected: FAIL with `ImportError` (cannot import `resolve_validation_strategy`).

- [ ] **Step 3: Implement `resolve_validation_strategy`**

In `src/mlops_agents/training/validation_policy.py`, add the constant near the other heuristics (after `_DEFAULT_N_FOLDS = 3`):

```python
_MAX_FOLDS = 5               # cap backtest folds regardless of how much history is available
```

And add the function directly below `select_validation_strategy`:

```python
def resolve_validation_strategy(task_metadata: dict[str, Any], n_obs: int) -> ValidationStrategy:
    """Capacity-driven validation strategy. Fold count scales with available history,
    floors at a single split, and hard-errors when even one clean split doesn't fit.

    `n_obs` is the full series length; the held-out test split takes the last `horizon`
    rows, so the validation budget is `n_obs - horizon`.
    """
    horizon = int(task_metadata["forecast_horizon"])
    drift = task_metadata.get("expected_drift", "low")
    train_pool_len = n_obs - horizon
    min_train = max(_HORIZON_MULTIPLIER * horizon, _MIN_TRAIN_ROWS)
    k_max = (train_pool_len - min_train) // horizon
    if k_max < 1:
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

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_training/test_validation_policy.py -k resolve -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/mlops_agents/training/validation_policy.py tests/test_training/test_validation_policy.py
git commit -m "feat(forecasting): capacity-driven resolve_validation_strategy"
```

---

## Task 2: `resolve_exog_strategies` (per-column, numeric-guarded)

**Files:**
- Modify: `src/mlops_agents/training/profiler.py` (public wrapper)
- Create: `src/mlops_agents/training/exog_policy.py`
- Test: `tests/test_training/test_exog_policy.py`

Context: `profiler.py` has private `def _detect_per_series(series: pd.Series, freq: str) -> tuple[bool, bool, bool]` returning `(seasonality, trend, stationarity)`. Expose a thin public wrapper rather than importing the private name across modules. `ExogStrategySettings` lives in `mlops_agents.contracts.training`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_training/test_exog_policy.py`:

```python
import numpy as np
import pandas as pd

from mlops_agents.training.exog_policy import resolve_exog_strategies


def _df(**series):
    n = len(next(iter(series.values())))
    base = {"ds": pd.date_range("2023-01-02", periods=n, freq="W-MON")}
    base.update(series)
    return pd.DataFrame(base)


def _meta(cols):
    return {"exogenous_columns": cols, "forecast_horizon": 8}


def test_seasonal_unknown_future_uses_ets():
    t = np.arange(156)
    temp = 9 - 13 * np.cos(2 * np.pi * (t % 52) / 52)   # strong yearly seasonality
    out = resolve_exog_strategies(
        _df(temp=temp), _meta([{"name": "temp", "future_availability": "unknown_future"}]), "W"
    )
    assert out.per_column["temp"] == "ets"


def test_flat_noise_unknown_future_uses_naive_carry():
    noise = np.random.default_rng(42).normal(0, 1, 156)
    out = resolve_exog_strategies(
        _df(noise=noise), _meta([{"name": "noise", "future_availability": "unknown_future"}]), "W"
    )
    assert out.per_column["noise"] == "naive_carry"


def test_non_numeric_unknown_future_uses_naive_carry():
    cond = ["sunny", "rainy"] * 78  # 156 strings
    out = resolve_exog_strategies(
        _df(cond=cond), _meta([{"name": "cond", "future_availability": "unknown_future"}]), "W"
    )
    assert out.per_column["cond"] == "naive_carry"


def test_known_future_absent_from_per_column():
    out = resolve_exog_strategies(
        _df(holiday=np.zeros(156)), _meta([{"name": "holiday", "future_availability": "known_future"}]), "W"
    )
    assert "holiday" not in out.per_column


def test_no_exog_declared_is_empty():
    out = resolve_exog_strategies(_df(y=np.arange(156, dtype=float)), {"exogenous_columns": []}, "W")
    assert out.per_column == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_training/test_exog_policy.py -v`
Expected: FAIL with `ModuleNotFoundError: mlops_agents.training.exog_policy`.

- [ ] **Step 3: Add the public wrapper in `profiler.py`**

In `src/mlops_agents/training/profiler.py`, add directly after `_detect_per_series`:

```python
def detect_series_structure(series: pd.Series, freq: str | None) -> tuple[bool, bool, bool]:
    """Public wrapper for _detect_per_series: (seasonality, trend, stationarity)."""
    return _detect_per_series(series, freq or "")
```

- [ ] **Step 4: Create `exog_policy.py`**

Create `src/mlops_agents/training/exog_policy.py`:

```python
"""Deterministic per-column exogenous-extension policy.

Decides how each unknown-future exog column is projected forward, so the LLM
planner no longer chooses it. known_future columns use their actual future
values (handled downstream) and are omitted from the per-column map.
"""
from __future__ import annotations

import pandas as pd

from mlops_agents.contracts.training import ExogStrategySettings
from mlops_agents.training.profiler import detect_series_structure


def resolve_exog_strategies(
    df: pd.DataFrame, task_metadata: dict, freq: str | None
) -> ExogStrategySettings:
    declared = task_metadata.get("exogenous_columns") or []
    per_column: dict[str, str] = {}
    for entry in declared:
        col, avail = entry["name"], entry["future_availability"]
        if avail != "unknown_future" or col not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            per_column[col] = "naive_carry"   # user-declared exog dtype is not guaranteed numeric
            continue
        seasonal, trend, _ = detect_series_structure(df[col].astype(float), freq)
        per_column[col] = "ets" if (seasonal or trend) else "naive_carry"
    return ExogStrategySettings(per_column=per_column, default_unknown_future="naive_carry")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_training/test_exog_policy.py -v`
Expected: PASS (5 tests).

- [ ] **Step 6: Commit**

```bash
git add src/mlops_agents/training/profiler.py src/mlops_agents/training/exog_policy.py tests/test_training/test_exog_policy.py
git commit -m "feat(forecasting): resolve_exog_strategies per-column extension policy"
```

---

## Task 3: Route the executor fallback through the policies

**Files:**
- Modify: `src/mlops_agents/training/executor.py`
- Modify: `src/mlops_agents/training/validation_policy.py` (remove `select_validation_strategy`)
- Test: `tests/test_training/test_executor_forecasting.py`, `tests/test_training/test_validation_policy.py`

- [ ] **Step 1: Write the failing test (fallback routes to multi-fold)**

Append to `tests/test_training/test_executor_forecasting.py` (the `air_passengers_csv` fixture and `run_training_plan` import are already used in this file; reuse them):

```python
import json
from pathlib import Path


def test_executor_fallback_resolves_multifold_validation(air_passengers_csv, tmp_path, monkeypatch):
    """A plan with forecasting_settings=None must route through resolve_validation_strategy
    (capacity-driven), NOT a hardcoded single_split. air_passengers (~144 obs) at horizon 12
    => multi-fold expanding_window."""
    monkeypatch.setattr("mlops_agents.training.executor.settings.experience_pool_dir", tmp_path / "pool")
    from mlops_agents.contracts.training import TrainingPlan, TrainingPlanCandidate, TrialBudget
    from mlops_agents.training.executor import run_training_plan

    plan = TrainingPlan(
        problem_type="forecasting",
        candidates=[TrainingPlanCandidate(priority=1, model_key="seasonal_naive")],
        trial_budget=TrialBudget(total_trials=2, min_trials_per_candidate=2, max_trials_per_candidate=2),
    )  # NOTE: no forecasting_settings -> exercises the executor fallback
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
        mlflow_experiment="test-fallback-multifold",
        random_state=42,
    )
    rec = json.loads(Path(result.experience_record_path).read_text())
    assert rec["validation_strategy"]["type"] == "expanding_window"   # not single_split — the fix
    assert rec["validation_strategy"]["n_folds"] >= 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_training/test_executor_forecasting.py -k fallback_resolves -v`
Expected: FAIL — current fallback uses `select_validation_strategy`, which returns `single_split` for `short` history, so `type` is `single_split`, not `expanding_window`.

- [ ] **Step 3: Switch the executor fallback to the policies**

In `src/mlops_agents/training/executor.py`, the import block around line 39-43 currently contains:

```python
from mlops_agents.training.validation_policy import (
    resolve_rolling_window_size,
    select_validation_strategy,
    validate_forecasting_plan,
)
```

Change to:

```python
from mlops_agents.training.validation_policy import (
    resolve_rolling_window_size,
    resolve_validation_strategy,
    validate_forecasting_plan,
)
from mlops_agents.training.exog_policy import resolve_exog_strategies
```

Then the fallback block (currently):

```python
    # Resolve forecasting_settings before any candidate runs
    fs = plan.forecasting_settings
    if fs is None and plan.problem_type == "forecasting":
        fs = ForecastingSettings(
            validation_strategy=select_validation_strategy(profile, task_metadata),
            exog_strategies=ExogStrategySettings(),
        )
        plan = plan.model_copy(update={"forecasting_settings": fs})
```

becomes:

```python
    # Resolve forecasting_settings before any candidate runs (fallback for plans built
    # without the planner, e.g. the benchmark runner / direct-executor tests).
    fs = plan.forecasting_settings
    if fs is None and plan.problem_type == "forecasting":
        _full_df = pd.read_csv(processed_dataset_path)
        fs = ForecastingSettings(
            validation_strategy=resolve_validation_strategy(task_metadata, len(_full_df)),
            exog_strategies=resolve_exog_strategies(
                _full_df, task_metadata, task_metadata.get("frequency")
            ),
        )
        plan = plan.model_copy(update={"forecasting_settings": fs})
```

(`pd` and `ForecastingSettings` are already imported in `executor.py`. `ExogStrategySettings` may now be an unused import — if `ruff` flags it, remove it from the imports.)

- [ ] **Step 4: Remove `select_validation_strategy` and its tests**

In `src/mlops_agents/training/validation_policy.py`, delete the entire `select_validation_strategy` function (the `def select_validation_strategy(...)` block) and update the module docstring line that references it (change "select_validation_strategy: picks the right ValidationStrategy…" to "resolve_validation_strategy: picks the ValidationStrategy from capacity…").

In `tests/test_training/test_validation_policy.py`, remove the import of `select_validation_strategy` and the three tests that call it (the ones under the `# ─── select_validation_strategy ───` section). Keep the `validate_forecasting_plan` tests and the new `resolve_*` tests.

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_training/ -q -m "not integration"`
Expected: PASS, including the new `test_executor_fallback_resolves_multifold_validation`. Confirm no remaining import of `select_validation_strategy`:
Run: `uv run python -c "import mlops_agents.training.executor, mlops_agents.training.validation_policy; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 6: Commit**

```bash
git add src/mlops_agents/training/executor.py src/mlops_agents/training/validation_policy.py tests/test_training/test_executor_forecasting.py tests/test_training/test_validation_policy.py
git commit -m "feat(forecasting): executor fallback uses capacity-driven validation + exog policies"
```

---

## Task 4: Resolve policies in `planner_node` and inject before validation

**Files:**
- Modify: `src/mlops_agents/planning/prompts.py`
- Modify: `src/mlops_agents/planning/node.py`
- Test: `tests/test_planning/test_node.py`

- [ ] **Step 1: Add the optional policy summary to `format_planner_inputs`**

In `src/mlops_agents/planning/prompts.py`, replace the whole `format_planner_inputs` function with:

```python
def format_planner_inputs(
    dataset_profile: dict[str, Any],
    task_metadata: dict[str, Any],
    problem_type: str,
    forecasting_policy_summary: str | None = None,
) -> str:
    """Compact human-readable summary of inputs to seed the agent's reasoning."""
    policy = ""
    if forecasting_policy_summary:
        policy = (
            "\n\nDeterministic forecasting policy (FIXED — do NOT choose validation or "
            f"exog strategy; select models suited to it):\n{forecasting_policy_summary}"
        )
    return (
        f"problem_type: {problem_type}\n\n"
        f"task_metadata:\n{json.dumps(task_metadata, indent=2, default=str)}\n\n"
        f"dataset_profile:\n{json.dumps(dataset_profile, indent=2, default=str)}"
        f"{policy}\n\n"
        f"Use the tools to retrieve evidence, then produce the PlannerOutput."
    )
```

- [ ] **Step 2: Write the failing test (policy injection)**

Append to `tests/test_planning/test_node.py` (it already imports `patch`, `MagicMock`, `planner_node`, and defines `_make_output_for`):

```python
@patch("mlops_agents.planning.node._check_conflict_resolution_present_if_flagged")
@patch("mlops_agents.planning.node._check_evidence_references_hybrid")
@patch("mlops_agents.planning.node._check_plan_exhaustiveness")
@patch("mlops_agents.planning.node._check_plan_integrity")
@patch("mlops_agents.planning.node.build_planner_validation_context")
@patch("mlops_agents.planning.node.build_planner_agent")
@patch("mlops_agents.planning.node.build_planner_tools")
@patch("mlops_agents.planning.node.build_dataset_profile")
def test_planner_node_injects_policy_forecasting_settings(
    mock_profile, mock_build_tools, mock_build_agent, mock_build_ctx,
    mock_integrity, mock_exhaust, mock_evidence, mock_conflict, tmp_path,
):
    """The plan's forecasting_settings must equal the deterministic policy output,
    regardless of what the LLM returns (here: forecasting_settings=None)."""
    mock_profile_instance = MagicMock()
    mock_profile_instance.model_dump.return_value = {}
    mock_profile.return_value = mock_profile_instance
    mock_ctx = MagicMock()
    mock_ctx.problem_type = "forecasting"
    mock_ctx.task_metadata = {}
    mock_ctx.available_model_keys = ["ets"]
    mock_ctx.similar_experiences = []
    mock_ctx.matched_rules = []
    mock_ctx.rules_by_id = {}
    mock_build_ctx.return_value = mock_ctx
    mock_build_tools.return_value = []
    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {
        "structured_response": _make_output_for("forecasting"),  # plan.forecasting_settings is None
        "messages": [],
    }
    mock_build_agent.return_value = fake_agent

    # 60 observations, horizon 8 -> capacity policy => expanding_window, 2 folds (NOT single_split)
    rows = "\n".join(f"2023-{(i % 12) + 1:02d}-01,{i}" for i in range(60))
    csv = tmp_path / "p.csv"
    csv.write_text("ds,y\n" + rows + "\n")
    state = {
        "processed_dataset_path": str(csv),
        "problem_type": "forecasting",
        "task_metadata": {
            "target_column": "y", "datetime_column": "ds",
            "forecast_horizon": 8, "exogenous_columns": [],
        },
    }
    result = planner_node(state)
    fs = result.update["training_plan"]["forecasting_settings"]
    assert fs is not None
    assert fs["validation_strategy"]["type"] == "expanding_window"
    assert fs["validation_strategy"]["n_folds"] == 2
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_planning/test_node.py -k injects_policy -v`
Expected: FAIL — currently `forecasting_settings` stays `None` (the LLM mock returned None and the node doesn't inject), so `assert fs is not None` fails.

- [ ] **Step 4: Wire the policies into `planner_node`**

In `src/mlops_agents/planning/node.py`, add imports near the top (with the other imports):

```python
import pandas as pd
from mlops_agents.contracts.training import ForecastingSettings
from mlops_agents.training.validation_policy import resolve_validation_strategy
from mlops_agents.training.exog_policy import resolve_exog_strategies
```

After `system_prompt = get_prompt("planner").template` and before `output = None`, insert:

```python
    # Deterministic forecasting policy: resolve validation + exog BEFORE the agent so the
    # LLM plans models under fixed settings it cannot override.
    forecasting_fs: ForecastingSettings | None = None
    policy_summary: str | None = None
    if problem_type == "forecasting":
        _df = pd.read_csv(processed_path)
        forecasting_fs = ForecastingSettings(
            validation_strategy=resolve_validation_strategy(task_meta, len(_df)),
            exog_strategies=resolve_exog_strategies(_df, task_meta, task_meta.get("frequency")),
        )
        policy_summary = (
            f"validation_strategy: type={forecasting_fs.validation_strategy.type}, "
            f"n_folds={forecasting_fs.validation_strategy.n_folds}; "
            f"exog extension per unknown-future column: "
            f"{forecasting_fs.exog_strategies.per_column or 'none'}"
        )
```

In the messages list, change the `HumanMessage` line to pass the summary:

```python
            HumanMessage(content=format_planner_inputs(profile, task_meta, problem_type, policy_summary)),
```

After `output = result.get("structured_response")` and its `if output is None: raise ...` block, and BEFORE `_check_plan_integrity(output, ...)`, insert:

```python
            if forecasting_fs is not None:
                output = output.model_copy(
                    update={"plan": output.plan.model_copy(
                        update={"forecasting_settings": forecasting_fs})}
                )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_planning/test_node.py -k injects_policy -v`
Expected: PASS.

- [ ] **Step 6: Run the planner + planning suites**

Run: `uv run pytest tests/test_planning/ tests/test_agents/test_planner_node.py -q -m "not integration"`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/mlops_agents/planning/prompts.py src/mlops_agents/planning/node.py tests/test_planning/test_node.py
git commit -m "feat(forecasting): resolve validation+exog policy in planner_node, inject before validation"
```

---

## Task 5: Remove obsolete planner-facing guidance

**Files:**
- Modify: `src/mlops_agents/knowledge/ml_rules.yaml`
- Modify: `src/mlops_agents/prompts/planner.yaml`
- Test: any rule-matching/planner test referencing the deleted rules

- [ ] **Step 1: Delete the 6 validation/exog rules**

In `src/mlops_agents/knowledge/ml_rules.yaml`, delete the entire block under the comment `# ─── forecasting_rules — exog & validation strategy guidance (SP5 planner) ─` — i.e. these 6 rule entries (and the comment line): `forecasting_short_history_single_split`, `forecasting_medium_long_expanding_window`, `forecasting_high_drift_rolling_window`, `exog_calendar_known_future`, `exog_unknown_default_naive_carry`, `exog_slow_macro_auto_arima`. Leave all the model-selection rules above them intact (e.g. `forecasting_short_history_prefers_statistical`, `forecasting_strong_seasonality_prefers_seasonal_models`).

- [ ] **Step 2: Rewrite the planner.yaml forecasting guidance (keep model-eligibility, drop validation/exog)**

In `src/mlops_agents/prompts/planner.yaml`, replace the `## Forecasting-specific guidance` block (currently):

```yaml
  ## Forecasting-specific guidance
  - If history_length is very_short or short: prefer single_split validation; include
    statistical baselines (naive, seasonal_naive, ets, auto_arima where registered);
    avoid high-complexity supervised models unless similar experiences strongly support them.
  - If expected_drift is high and history is sufficient: prefer rolling_window.
  - For unknown_future exogenous columns: choose extension strategies from
    {naive_carry, ets, auto_arima}.
  - known_future variables must not appear in per-column unknown-future overrides at all.
```

with:

```yaml
  ## Forecasting-specific guidance
  - If history_length is very_short or short: include statistical baselines (naive,
    seasonal_naive, ets, auto_arima where registered); avoid high-complexity supervised
    models unless similar experiences strongly support them.
  - Validation strategy and exogenous-extension handling are FIXED by deterministic
    policy (shown in your inputs) — do not choose them; select models suited to them.
```

- [ ] **Step 3: Update any tests referencing the deleted rules**

Run: `uv run grep -rn "forecasting_short_history_single_split\|forecasting_medium_long_expanding_window\|forecasting_high_drift_rolling_window\|exog_calendar_known_future\|exog_unknown_default_naive_carry\|exog_slow_macro_auto_arima" tests/ src/`
For any test that asserts these rules match/exist, remove that assertion or the test. (The rule-reader and `match_rules` are data-driven, so deleting the YAML entries needs no code change — only tests that hardcoded those rule_ids.)

- [ ] **Step 4: Run the knowledge + planning suites**

Run: `uv run pytest tests/test_planning/ tests/test_knowledge/ -q -m "not integration"` (run whichever of these dirs exist)
Expected: PASS. Then a YAML sanity check:
Run: `uv run python -c "from mlops_agents.knowledge.reader import match_rules; print('rules load ok')"`
Expected: prints `rules load ok`.

- [ ] **Step 5: Commit**

```bash
git add src/mlops_agents/knowledge/ml_rules.yaml src/mlops_agents/prompts/planner.yaml tests/
git commit -m "refactor(forecasting): drop planner-facing validation/exog guidance (now deterministic)"
```

---

## Task 6: Full-suite verification

**Files:** none (verification only).

- [ ] **Step 1: Run the whole unit suite**

Run: `uv run pytest -q -m "not integration"`
Expected: PASS. Investigate any failure referencing validation strategy, exog, or planner output.

- [ ] **Step 2: Type-check the touched modules**

Run: `uv run mypy src/mlops_agents/training/validation_policy.py src/mlops_agents/training/exog_policy.py src/mlops_agents/training/profiler.py src/mlops_agents/planning/node.py src/mlops_agents/planning/prompts.py`
Expected: no new errors.

- [ ] **Step 3: Lint**

Run: `uv run ruff check src/mlops_agents/training/ src/mlops_agents/planning/`
Expected: clean (fix any unused-import warning, e.g. a now-unused `ExogStrategySettings` in `executor.py`).

---

## Self-Review

**1. Spec coverage**
- Capacity-driven validation, hard-error floor → Task 1.
- Per-column profiled exog (ets/naive_carry) + non-numeric guard → Task 2.
- Shared policy functions for both call sites → Task 1/2 (definitions) + Task 3 (executor) + Task 4 (planner).
- Resolve-before-planner + inject-before-validation ordering → Task 4 Step 4.
- LLM emits models only; remove 6 rules + planner.yaml guidance (keep model-eligibility) → Task 5.
- `_check_plan_integrity` step 5 kept as guard → unchanged (validates injected policy settings); Task 4 injects before it runs.
- Experience pool unchanged → no task (schema untouched).
- Testing: ValidationPolicy (Task 1), ExogPolicy incl. non-numeric (Task 2), executor fallback (Task 3), planner_node injection (Task 4).

**2. Placeholder scan:** none — every code step has full content; the only "search and adapt" step (Task 5 Step 3) is a mechanical grep-and-remove of hardcoded rule_ids, with the exact rule_ids given.

**3. Type consistency:** `resolve_validation_strategy(task_metadata: dict, n_obs: int) -> ValidationStrategy` and `resolve_exog_strategies(df, task_metadata, freq) -> ExogStrategySettings` are used with the same signatures at all three call sites (Task 3 executor, Task 4 planner). `detect_series_structure(series, freq: str | None)` matches the `_detect_per_series` return tuple. `ForecastingSettings(validation_strategy=…, exog_strategies=…)` matches the contract. The planner injection uses `model_copy` on both `PlannerOutput` and its `.plan` consistently.
