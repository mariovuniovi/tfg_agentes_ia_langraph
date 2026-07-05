# Foundation: ML Dataset Contract & State Model

**Date:** 2026-05-04
**Status:** Approved
**Branch:** feature/ml-dataset-contract

## Problem

The pipeline currently supports classification only. The schema has no `problem_type` declaration, `AgentState` carries no task-level metadata, and the trainer/evaluator operate with hardcoded classification assumptions. Adding regression and time-series forecasting (multi-series, panel data) requires a shared contract — declared in the schema, propagated through state, and consumed by all downstream agents.

This is sub-project 1 of 4. It establishes the foundation everything else builds on:
- Sub-project 2: Forecasting-aware data validator/preprocessor
- Sub-project 3: Trainer strategy dispatch
- Sub-project 4: Evaluator strategy dispatch

## Design Principles

- **Declare once, propagate automatically** — `problem_type` and task metadata are written to state once by `data_validator_node` (deterministically, no LLM), then read by all downstream context builders.
- **Schema is the contract** — top-level schema fields are the single source of truth for task type and column roles. No duplication into column-level fields.
- **Fail fast, no silent defaults** — missing or invalid schema fields raise an error immediately. A missing `problem_type` is an error, not a fallback to classification. A missing `forecast_horizon` is an error, not a default of 1.
- **No LLM for metadata extraction or validation** — parsing, validating, and writing to state is pure Python.

## Schema Format (ML Dataset Contract)

The schema JSON gains top-level task declaration fields alongside the existing `columns` array.

### Classification / Regression schema
```json
{
  "problem_type": "classification",
  "target_column": "species",
  "columns": [
    {"name": "sepal_length", "dtype": "float", "nullable": false, "description": "Sepal length in cm"},
    {"name": "species",      "dtype": "string", "nullable": false, "is_key": true, "description": "Iris species label"}
  ]
}
```

### Forecasting schema
```json
{
  "problem_type": "forecasting",
  "datetime_column": "date",
  "target_column": "sales",
  "series_id_columns": ["product_id", "store_id"],
  "forecast_horizon": 30,
  "frequency": "D",
  "columns": [
    {"name": "date",       "dtype": "datetime", "nullable": false, "description": "Transaction date"},
    {"name": "product_id", "dtype": "string",   "nullable": false, "description": "Product identifier"},
    {"name": "store_id",   "dtype": "string",   "nullable": false, "description": "Store identifier"},
    {"name": "sales",      "dtype": "float",    "nullable": true,  "description": "Units sold"},
    {"name": "price",      "dtype": "float",    "nullable": true,  "description": "Sale price"}
  ]
}
```

### Top-level fields

| Field | Required for | Type | Notes |
|---|---|---|---|
| `problem_type` | all | string | `"classification"` \| `"regression"` \| `"forecasting"` — **no default, must be declared** |
| `target_column` | all | string | Must match a column name in `columns` |
| `datetime_column` | forecasting | string | Must match a column name in `columns` |
| `series_id_columns` | forecasting | list of strings | Each entry must match a column name; may be `[]` for single-series |
| `forecast_horizon` | forecasting | positive integer | **No default** — must be declared explicitly |
| `frequency` | forecasting | string | pandas offset alias (`"D"`, `"W"`, `"M"`, `"H"`, etc.) — **no default** |

### Column-level fields (unchanged)

Existing fields remain as-is: `name`, `dtype`, `nullable`, `description`, `min`, `max`, `allowed_values`, `mapping_hints`, `is_key`. No new column-level fields are added. Column roles (datetime, identifier, target, exogenous) are derived by the pipeline from top-level fields at runtime.

### Deriving column roles from top-level fields

| Condition | Derived role | Imputation policy (sub-project 2) |
|---|---|---|
| `col.name == datetime_column` | datetime | No imputation — flag missing timestamps |
| `col.name in series_id_columns` | identifier | No imputation |
| `col.name == target_column` | target | Context-dependent (sub-project 2) |
| All others (forecasting) | exogenous | Forward fill / interpolation |
| All others (regression/classification) | feature | Existing mean/median/mode |

## Contract Validation Rules

Validated deterministically in `data_validator_node` before the agent runs. Any violation raises a `ValueError` that sets `error_message` and routes immediately to FINISH — no LLM call is made.

**All problem types:**
- `problem_type` must be present and one of `"classification"`, `"regression"`, `"forecasting"`. Missing or unrecognised value → error.
- `target_column` must be present and must exist in `columns`.

**Forecasting only:**
- `datetime_column` must be present and must exist in `columns`.
- `forecast_horizon` must be present and a positive integer. Missing or ≤ 0 → error.
- `frequency` must be present and a non-empty string. Missing or empty → error.
- Every entry in `series_id_columns` must exist in `columns`.

**Regression only:**
- `target_column` dtype should be numeric (`float` or `int`). Non-numeric target → warning logged, not error (the agent may handle it).

**Classification only:**
- `target_column` dtype should be categorical, boolean, string, or discrete integer. Numeric continuous target → warning logged, not error.

Validation code pattern:

```python
def _validate_schema_contract(schema_data: dict) -> None:
    column_names = {c["name"] for c in schema_data.get("columns", [])}

    problem_type = schema_data.get("problem_type")
    if problem_type not in ("classification", "regression", "forecasting"):
        raise ValueError(
            f"Schema missing or invalid 'problem_type'. Got: {problem_type!r}. "
            "Must be 'classification', 'regression', or 'forecasting'."
        )

    target_column = schema_data.get("target_column")
    if not target_column or target_column not in column_names:
        raise ValueError(
            f"'target_column' must be declared and exist in columns. Got: {target_column!r}."
        )

    if problem_type == "forecasting":
        required = ["datetime_column", "forecast_horizon", "frequency"]
        missing = [f for f in required if not schema_data.get(f)]
        if missing:
            raise ValueError(f"Forecasting schema missing required fields: {missing}")

        if not isinstance(schema_data["forecast_horizon"], int) or schema_data["forecast_horizon"] <= 0:
            raise ValueError(
                f"'forecast_horizon' must be a positive integer. Got: {schema_data['forecast_horizon']!r}."
            )

        if schema_data["datetime_column"] not in column_names:
            raise ValueError(
                f"'datetime_column' '{schema_data['datetime_column']}' not found in columns."
            )

        for col in schema_data.get("series_id_columns", []):
            if col not in column_names:
                raise ValueError(f"'series_id_columns' entry '{col}' not found in columns.")
```

## Deterministic Metadata Extraction in `data_validator_node`

Two separate steps — validation and extraction are distinct concerns:

**Step 1 — Contract validation (before the agent):** `_validate_schema_contract` runs before `agent.invoke`. If the schema is structurally invalid, the node returns immediately with `error_message` set and no LLM call is made. The agent does not need the extracted fields — it already receives the full `schema_json` via `_build_data_validator_context` and reads `problem_type`, `datetime_column`, etc. directly from it.

**Step 2 — Metadata extraction (after the agent):** Once the agent has run successfully and `dataset_summary` is built, extract `problem_type` and `task_metadata` from `schema_data` into typed state fields. These exist for downstream nodes (trainer, evaluator, deployer) so they can read clean typed values from state rather than re-parsing the schema JSON themselves.

```python
import json as _json

schema_data = _json.loads(schema_json) if schema_json != "{}" else {}

try:
    _validate_schema_contract(schema_data)
except ValueError as exc:
    error_msg = f"Schema contract violation: {exc}"
    logger.error(f"[data_validator] {error_msg}")
    return Command(
        update={
            "messages": [HumanMessage(content=error_msg, name="data_validator")],
            "validation_passed": False,
            "error_message": error_msg,
            "problem_type": "",
            "task_metadata": {},
            "dataset_summary": {},
            "validation_report": {},
            "dataset_path": "",
        },
        goto="supervisor",
    )

problem_type = schema_data["problem_type"]
task_metadata: dict = {"target_column": schema_data["target_column"]}
if problem_type == "forecasting":
    task_metadata.update({
        "datetime_column": schema_data["datetime_column"],
        "series_id_columns": schema_data.get("series_id_columns", []),
        "forecast_horizon": schema_data["forecast_horizon"],
        "frequency": schema_data["frequency"],
    })
```

`problem_type` and `task_metadata` are then written to state via `Command.update` in every return path alongside `dataset_summary`.

## AgentState Additions

Two new fields in `src/mlops_agents/state/agent_state.py`:

```python
# Task type — written once by data_validator_node before agent invocation
problem_type: str   # "classification" | "regression" | "forecasting"

# Task-level metadata — written once by data_validator_node
task_metadata: dict
# classification/regression: {"target_column": str}
# forecasting: {
#   "target_column": str,
#   "datetime_column": str,
#   "series_id_columns": list[str],
#   "forecast_horizon": int,
#   "frequency": str,
# }
```

Both initialized to `""` and `{}` respectively in `main()` initial state.

## Context Builder Updates

### `_build_data_validator_context` — no change

Already passes the full `schema_json` to the agent. The agent sees `problem_type`, `datetime_column`, etc. as part of the schema it receives.

### `_build_trainer_context` — two new lines

```python
def _build_trainer_context(state: AgentState) -> HumanMessage:
    return HumanMessage(content=(
        f"Problem type: {state.get('problem_type', '')}\n"
        f"Task metadata: {json.dumps(state.get('task_metadata') or {})}\n"
        f"Canonical dataset: {state.get('dataset_path', '')}\n"
        f"Dataset summary: {json.dumps(state.get('dataset_summary') or {})}"
    ))
```

### `_build_evaluator_context` — two new lines

```python
def _build_evaluator_context(state: AgentState) -> HumanMessage:
    return HumanMessage(content=(
        f"Problem type: {state.get('problem_type', '')}\n"
        f"Task metadata: {json.dumps(state.get('task_metadata') or {})}\n"
        f"Training run ID: {state.get('training_run_id', '')}\n"
        f"Trained model path: {state.get('trained_model_path', '')}\n"
        f"Training metrics: {json.dumps(state.get('training_metrics') or {})}"
    ))
```

### `_build_deployer_context` — one new line

```python
def _build_deployer_context(state: AgentState) -> HumanMessage:
    return HumanMessage(content=(
        f"Problem type: {state.get('problem_type', '')}\n"
        f"Best model URI: {state.get('best_model_uri', '')}\n"
        f"Training run ID: {state.get('training_run_id', '')}\n"
        f"Evaluation report: {json.dumps(state.get('evaluation_report') or {})}"
    ))
```

## Supervisor Snapshot Update

Add `problem_type` to the structured snapshot in `supervisor.py`:

```python
snapshot_data = {
    "problem_type": state.get("problem_type", ""),
    "validation_passed": state.get("validation_passed") if dv_has_run else None,
    "evaluation_passed": state.get("evaluation_passed"),
    "deployment_decision": state.get("deployment_decision", "pending"),
    "error_message": state.get("error_message", ""),
    "training_run_id": state.get("training_run_id", ""),
}
```

## Files Changed

| File | Change |
|---|---|
| `src/mlops_agents/state/agent_state.py` | Add `problem_type: str` and `task_metadata: dict` |
| `src/mlops_agents/graphs/mlops_graph.py` | Add `_validate_schema_contract`; extract/validate metadata before agent call in `data_validator_node`; add `problem_type` and `task_metadata` to all `Command.update` return paths; update 4 context builders |
| `src/mlops_agents/agents/supervisor.py` | Add `problem_type` to snapshot |
| `data/schemas/iris_classification.json` | Add `problem_type: "classification"` and `target_column: "species"` |
| `tests/test_graphs/test_node_state_extraction.py` | Add `problem_type` and `task_metadata` to `_make_state`; add contract validation tests; add extraction tests |
| `tests/test_agents/test_supervisor.py` | Add `problem_type` and `task_metadata` to `make_state`; update snapshot test |

## What This Enables

- Sub-project 2 reads `problem_type` and `task_metadata` from state to apply time-aware imputation and temporal validation
- Sub-project 3 trainer reads `problem_type` from context to dispatch the correct training strategy
- Sub-project 4 evaluator reads `problem_type` from context to dispatch the correct evaluation strategy
- The graph topology does not change — supervisor routing is unaffected
- Invalid or incomplete schemas fail immediately with a clear error before any LLM call is made
