# Auto-Fix Validation HITL Design

**Date:** 2026-05-02  
**Status:** Approved  
**Branch:** feature/fastapi-backend

## Problem

The current data validation HITL gate fires regardless of whether validation passed or failed. When validation fails the agent has no tools to fix the data, so it falls back to asking the user conversational questions ("Shall I impute? Reply Yes to proceed"). Nobody can answer those questions — there is no chat interface. The result is an infinite supervisor → data_validator loop until `max_attempts_per_agent` is hit.

Two root causes:
1. Agent prompt does not forbid asking questions.
2. Node fires `interrupt()` even when `validation_passed=False`, giving the supervisor no clear signal to abort.

## Goals

- Agent autonomously fixes fixable data issues (nullable violations) using a config-driven strategy.
- HITL gate fires **only when validation passes** — human reviews a clean, verified dataset.
- If validation still fails after auto-fix, pipeline aborts with a clear error message, no HITL shown.
- If human rejects the validated dataset at HITL, pipeline aborts — retry cannot add value because tools and strategy are deterministic.
- Fix strategy is defined in config, not by the agent — the agent executes policy, it does not set it.

## Design

### 1. Config (`settings.py`)

Two new fields:

```python
imputation_strategy_numeric: Literal["mean", "median", "zero"] = "mean"
imputation_strategy_categorical: Literal["mode", "unknown", "drop_row"] = "mode"
```

Validated by Pydantic at startup. No other imputation parameters needed for now.

### 2. New tool: `impute_missing_values` (`data_tools.py`)

```python
impute_missing_values(path: str) -> dict
```

- Reads the CSV at `path`.
- For each column with NaN values: applies `settings.imputation_strategy_numeric` for numeric dtypes (`float64`, `int64`), `settings.imputation_strategy_categorical` for `object` dtype.
- Writes the imputed DataFrame back to the same path (in-place).
- Returns:
  ```json
  {
    "output_path": "data/processed/iris_classification.csv",
    "imputed_columns": {
      "sepal_width": {"strategy": "mean", "fill_value": 3.41, "rows_affected": 1}
    }
  }
  ```
- The agent passes only the file path. Strategy is read from `settings` inside the tool — the agent cannot override it.

### 3. Agent prompt (`data_agent.yaml`)

Two changes:

**Add `impute_missing_values` to the TOOLS section** with its description.

**Add an autonomy rule** (prominent, at the top):

> **AUTONOMY RULE: Never ask the user questions. Never ask for confirmation before calling a tool or modifying a file. Complete the full task and stop. The pipeline is fully automated — there is no one to answer your questions.**

**Extend PROCESS step 5** to include the imputation-then-revalidate loop:

> 5. Call `validate_against_schema` on the canonical output file.  
> 5b. If validation fails **only due to nullable violations**, call `impute_missing_values` on the canonical file, then call `validate_against_schema` again. Do this once. If validation still fails, report the remaining violations and stop — do not ask the user what to do.

### 4. `data_validator_node` (`mlops_graph.py`)

`interrupt()` is moved inside a conditional:

```python
if validation_passed:
    approval = interrupt({
        "type": "data_validation",
        "question": "Review the processed dataset before training begins.",
        "attempt": attempt,
        "dataset_preview": preview,
        "validation_summary": {...},
        "imputation_applied": imputation_result,  # new field
    })

    if approval.get("approved", False):
        return Command(update=base_update, goto="supervisor")

    # Human rejected a validated dataset — abort, no retry possible
    comment = approval.get("comment", "")
    rejection_text = f"Dataset rejected by human reviewer. Comment: {comment}" if comment else "Dataset rejected by human reviewer."
    return Command(
        update={**base_update, "validation_passed": False, "error_message": rejection_text},
        goto="supervisor",
    )

else:
    # Validation failed after auto-fix attempt — abort without HITL
    return Command(
        update={
            **base_update,
            "error_message": f"Data validation failed after auto-fix: {final_message}",
        },
        goto="supervisor",
    )
```

The `imputation_applied` field in the interrupt payload lets the HITL review panel show the user what was automatically fixed (e.g., "sepal_width: 1 row imputed with mean=3.41").

### 5. Supervisor prompt (`supervisor.yaml`)

Rule 5 is replaced with a more explicit version:

> 5. If `error_message` is set in state, always select FINISH — do not retry any agent. If `validation_passed=False` after `data_validator` has already run, select FINISH — imputation is handled automatically inside the agent, not by retrying.

## End-to-End Flow

```
Upload files → data_validator agent runs
  └─ load → merge → map → validate
       ├─ passes → HITL fires → human approves → proceed to trainer
       │                      → human rejects → FINISH (abort, no retry)
       └─ fails (nullable violations) → impute_missing_values → re-validate
            ├─ passes → HITL fires → human approves → proceed to trainer
            │                      → human rejects → FINISH (abort, no retry)
            └─ still fails (unfixable: type error, wrong values, missing column)
                 → error_message set, no HITL → supervisor selects FINISH
```

## What is NOT in scope

- Imputation of range violations or type mismatches — those require source data correction.
- Letting the human choose the imputation strategy at runtime — that is a config concern.
- Retry after human rejection — retrying is pointless because tools and strategy are deterministic.
- Multi-column or cross-column imputation strategies (e.g., k-NN) — out of scope for this iteration.

## Files Changed

| File | Change |
|------|--------|
| `src/mlops_agents/config/settings.py` | Add `imputation_strategy_numeric`, `imputation_strategy_categorical` |
| `src/mlops_agents/tools/data_tools.py` | Add `impute_missing_values` tool |
| `src/mlops_agents/agents/data_agent.py` | Register new tool |
| `src/mlops_agents/prompts/data_agent.yaml` | Add autonomy rule + imputation step to PROCESS |
| `src/mlops_agents/graphs/mlops_graph.py` | Move `interrupt()` inside `if validation_passed` |
| `src/mlops_agents/prompts/supervisor.yaml` | Strengthen rule 5 |
| `tests/test_tools/test_data_tools.py` | Tests for `impute_missing_values` |
| `tests/test_graphs/test_node_state_extraction.py` | Update node tests for new HITL behavior |
