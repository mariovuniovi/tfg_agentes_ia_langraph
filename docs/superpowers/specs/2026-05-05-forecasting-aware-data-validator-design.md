# Forecasting-Aware Data Validator/Preprocessor — Design Spec

**Sub-project:** 2 of the ML Dataset Contract foundation  
**Date:** 2026-05-05  
**Branch:** feature/ml-dataset-contract

---

## Goal

Extend the data validator to handle forecasting datasets correctly: role-aware imputation, temporal gap detection with a compact audit-friendly report, and critical forecasting key validation that aborts early if datetime or series IDs are broken.

---

## Architecture

The data validator already calls `impute_missing_values`, `validate_against_schema`, and `check_data_quality` as tools via the react agent. This sub-project adds two new deterministic tools (`parse_datetime_column`, `detect_temporal_gaps`) and restructures `impute_missing_values` into a role-aware dispatcher. No new AgentState fields are added. Gap reports are written to `artifacts/temporal_gaps.json` and the path is embedded in `validation_report`.

---

## Section 1 — Modified `impute_missing_values`

### Signature

```python
def impute_missing_values(
    dataset_path: str,
    problem_type: str,               # "classification" | "regression" | "forecasting"
    target_column: str,
    datetime_column: str | None = None,
    series_id_columns: list[str] = [],
    max_interpolation_gap: int = 3,  # max consecutive periods for target interpolation
    output_path: str = "",
) -> dict
```

### Behaviour

- No fallback for `problem_type` — raise `ValueError` immediately if not one of the three valid values
- Dispatch to `TabularImputer` for `"classification"` / `"regression"`, `ForecastingImputer` for `"forecasting"`

**`TabularImputer`** (classification / regression):
- Mean for numeric columns, mode for categoricals, including the target column
- Returns `{"output_path": ..., "columns_imputed": [...], "rows_affected": int}`

**`ForecastingImputer`** (forecasting):
- `datetime_column` and all `series_id_columns`: protected — if any null value exists, raise immediately (never impute)
- `target_column`: short-gap linear interpolation only (gap ≤ `max_interpolation_gap` consecutive periods); gaps larger than the threshold are flagged in the return value as warnings, never silently filled
- All other columns (exogenous features): forward-fill followed by linear interpolation
- Returns `{"output_path": ..., "columns_imputed": [...], "rows_affected": int, "target_large_gaps": [{"series_id": {...}, "gap_size": int}]}`

---

## Section 2 — Two New Deterministic Tools

### `parse_datetime_column`

```python
def parse_datetime_column(
    dataset_path: str,
    datetime_col: str,
    output_path: str = "",
) -> dict
```

- Parses the column with `pd.to_datetime`, raises `ValueError` if parsing fails or the column contains any nulls after parsing
- Sorts the dataset by `datetime_col` (and `series_id_columns` if provided)
- Writes sorted dataset to `output_path` (or overwrites `dataset_path` if empty)
- Returns `{"output_path": ..., "dtype": "datetime64", "null_count": 0}`

### `detect_temporal_gaps`

```python
def detect_temporal_gaps(
    dataset_path: str,
    datetime_col: str,
    series_id_cols: list[str],
    frequency: str,           # pandas offset alias, e.g. "D", "W", "MS"
    target_column: str,
    output_path: str = "",    # path to write full gap report JSON artifact
) -> dict
```

**Critical key validation** (runs first, raises if any fail):
- `datetime_col` column exists and has no nulls
- All `series_id_cols` columns exist and have no nulls
- `target_column` exists in the dataset

**Gap detection:**
- Per series (grouped by `series_id_cols`), generate the expected date range from min to max at `frequency`
- Compare against actual dates to find missing periods

**Return format** (compact — no massive lists):
```python
{
    "has_gaps": True,
    "total_missing_periods": 248,
    "n_series_with_gaps": 17,
    "gap_examples": [
        {
            "series_id": {"product_id": "P01", "store_id": "S03"},
            "n_missing_periods": 12,
            "first_missing": "2025-01-08",
            "last_missing": "2025-02-14",
            "sample_missing_dates": ["2025-01-08", "2025-01-09", "2025-01-15"]
        }
        # up to 5 worst-offending series
    ],
    "artifact_path": "artifacts/temporal_gaps.json"
}
```

The full gap report (all series, all missing dates) is written to `output_path` / `artifacts/temporal_gaps.json` as a JSON artifact for auditability. `detect_temporal_gaps` also detects duplicate (series_id, datetime) pairs and raises if any are found.

---

## Section 3 — Agent Tool Call Order (Forecasting Branch)

The `data_agent.yaml` prompt instructs the agent to follow this sequence for `problem_type == "forecasting"`:

1. `parse_datetime_column` — parse and sort; abort if nulls or parse failure
2. `detect_temporal_gaps` — validate critical keys, detect gaps and duplicates; abort if critical keys broken
3. `impute_missing_values` — role-aware; `max_interpolation_gap` must be passed explicitly
4. `validate_against_schema` — Pydantic contract check
5. `check_data_quality` — Evidently summary

For `"classification"` / `"regression"`, the existing two-step flow (`impute_missing_values` → `validate_against_schema` → `check_data_quality`) continues unchanged.

Imputation prompt wording: "time-aware imputation — protected datetime/series IDs (raise if null), cautious target short-gap interpolation only (≤ max_interpolation_gap consecutive periods), exogenous forward-fill/linear interpolation."

---

## Section 4 — HITL Payload and Audit Trail

The gap report summary is embedded in the `interrupt()` HITL payload so the human reviewer sees it:

```python
interrupt({
    "type": "deployer",
    ...,
    "temporal_gaps": detect_temporal_gaps_result,   # compact summary dict
})
```

The full gap report path is embedded in `validation_report` (an existing state field — no new top-level fields):

```python
validation_report = {
    ...,
    "temporal_gaps_report_path": "artifacts/temporal_gaps.json"
}
```

This ensures that when a human approves a deployment, the audit trail records that they approved a dataset that may have had temporal gaps.

---

## Section 5 — Scope

### Files modified

| File | Change |
|---|---|
| `src/mlops_agents/tools/data_tools.py` | Restructure `impute_missing_values` into role-aware dispatcher; add `parse_datetime_column` and `detect_temporal_gaps` |
| `src/mlops_agents/prompts/data_agent.yaml` | Add forecasting branch with explicit tool call order and imputation wording |
| `src/mlops_agents/agents/data_agent.py` | Register the two new tools so the agent can call them |
| `tests/test_tools/test_data_tools.py` | Tests for all new/modified tool behaviour (TDD) |

### Not in scope

- No new AgentState top-level fields
- No changes to supervisor, trainer, evaluator, or deployer nodes
- No changes to frontend or API layer
- Sub-project 3 (trainer strategy dispatch) is separate
