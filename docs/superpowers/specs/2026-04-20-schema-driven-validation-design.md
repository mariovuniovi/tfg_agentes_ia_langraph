# Schema-Driven Data Validation Design

**Date:** 2026-04-20
**Status:** Approved

## Problem

The current data validation agent only checks that a `target` column is present and runs generic Evidently quality checks. It has no knowledge of the canonical dataset shape, cannot map raw column names to expected ones, and produces no cleaned output file for downstream nodes.

## Goal

Replace the generic validation with a schema-driven approach: a committed JSON schema defines the canonical dataset (columns, types, constraints, mapping hints). The data validation agent reads the raw CSV, uses the schema — injected into its initial message — to map raw columns to canonical ones, writes a cleaned CSV, and then validates it against all constraints.

---

## Schema Format

Schema files live at `data/schemas/<name>.json`. The active schema is selected via `settings.dataset_schema` (default: `"iris_classification"`).

```json
{
  "name": "iris_classification",
  "description": "Iris flower classification dataset",
  "columns": [
    {
      "name": "sepal_length",
      "dtype": "float",
      "description": "Sepal length in cm",
      "required": true,
      "nullable": false,
      "min": 0.0,
      "max": 30.0,
      "mapping_hint": "Usually named 'SepalLengthCm', 'sepal length (cm)' or similar in raw files"
    },
    {
      "name": "sepal_width",
      "dtype": "float",
      "description": "Sepal width in cm",
      "required": true,
      "nullable": false,
      "min": 0.0,
      "max": 30.0,
      "mapping_hint": "Usually named 'SepalWidthCm', 'sepal width (cm)' or similar"
    },
    {
      "name": "petal_length",
      "dtype": "float",
      "description": "Petal length in cm",
      "required": true,
      "nullable": false,
      "min": 0.0,
      "max": 30.0,
      "mapping_hint": "Usually named 'PetalLengthCm', 'petal length (cm)' or similar"
    },
    {
      "name": "petal_width",
      "dtype": "float",
      "description": "Petal width in cm",
      "required": true,
      "nullable": false,
      "min": 0.0,
      "max": 30.0,
      "mapping_hint": "Usually named 'PetalWidthCm', 'petal width (cm)' or similar"
    },
    {
      "name": "target",
      "dtype": "str",
      "description": "Class label for the flower species",
      "required": true,
      "nullable": false,
      "unique": false,
      "allowed_values": ["setosa", "versicolor", "virginica"],
      "mapping_hint": "Often named 'species', 'class', 'label', or 'variety' in raw datasets"
    }
  ]
}
```

### Supported column fields

| Field | Type | Required | Meaning |
|-------|------|----------|---------|
| `name` | string | yes | Canonical column name in the output CSV |
| `dtype` | `"float"` \| `"int"` \| `"str"` \| `"bool"` | yes | Expected pandas dtype family |
| `description` | string | yes | Human-readable description |
| `required` | bool | yes | Whether the column must be present |
| `nullable` | bool | yes | Whether null values are allowed |
| `unique` | bool | no | Whether all values must be distinct |
| `min` | number | no | Minimum numeric value (inclusive) |
| `max` | number | no | Maximum numeric value (inclusive) |
| `allowed_values` | list | no | Exhaustive set of permitted values |
| `mapping_hint` | string | no | Free-text hint for the LLM to recognise this column in raw data |

---

## Data Flow

```
data_validator_node (mlops_graph.py)
  │
  ├─ reads settings.dataset_schema  →  data/schemas/<name>.json
  ├─ builds HumanMessage:
  │    "Raw dataset: <path>\nTarget schema:\n<schema JSON>"
  │
  └─ invokes data_agent (ReAct loop)
       │
       ├─ load_dataset          [existing] inspect raw columns/types
       ├─ apply_column_mapping  [new] rename/drop/reorder → write canonical CSV
       ├─ validate_against_schema [new] check all constraints → pass/fail report
       └─ check_data_quality    [existing, optional] Evidently quality report
```

After a successful run, `data_validator_node` updates `AgentState.dataset_path` to point to the canonical CSV (`data/processed/<name>.csv`) so training uses the clean file.

---

## Components

### `settings.dataset_schema` (config/settings.py)

New field with default `"iris_classification"`. No `.env` change required for the demo.

```python
dataset_schema: str = "iris_classification"
```

### `apply_column_mapping` (tools/data_tools.py) — NEW

```
apply_column_mapping(raw_path, mapping_json, output_path) -> str
```

- `mapping_json`: JSON object `{"raw_col": "canonical_col", ...}` produced by the LLM
- Renames columns per the mapping, drops any column not in the schema, writes to `output_path`
- Returns JSON with `{success, output_path, mapped_columns, dropped_columns}`
- Errors gracefully if a required canonical column ends up missing after mapping

### `validate_against_schema` (tools/data_tools.py) — NEW

```
validate_against_schema(canonical_path, schema_name) -> str
```

- Loads `data/schemas/<schema_name>.json`
- Checks every constraint deterministically: `required`, `nullable`, `unique`, `min`/`max`, `allowed_values`, dtype compatibility
- Returns JSON with `{passed, violations: [{column, rule, detail}, ...]}`

### `validate_schema` (tools/data_tools.py) — REMOVED

Replaced by `validate_against_schema`. Existing tests for `validate_schema` are replaced.

---

## Agent Prompt (`prompts/data_agent.yaml`)

Updated workflow section:

1. Call `load_dataset` to inspect the raw column names and types.
2. Use the schema (already in context) and its `mapping_hint` fields to decide the column mapping. Produce a JSON object `{"raw_col": "canonical_col"}` for every required column you can match. Report any columns you cannot map.
3. Call `apply_column_mapping` with your mapping decision and `data/processed/<schema_name>.csv` as the output path.
4. Call `validate_against_schema` on the canonical output file.
5. Optionally call `check_data_quality` for an Evidently summary.
6. Report: PASSED or FAILED, which columns were mapped from what, any violations.

---

## Testing

### `apply_column_mapping`
- Correct rename: raw `{"species": "target"}` → canonical column `target` present
- Drops columns not in schema
- Writes output CSV to the specified path
- Returns error JSON (not exception) when a required column is missing after mapping

### `validate_against_schema`
- `nullable=false` violated → violation reported
- `unique=true` violated → violation reported
- `min`/`max` out of range → violation reported
- `allowed_values` violated → violation reported
- Missing required column → violation reported
- All constraints satisfied → `passed: true`, empty violations list

### Removed
- `test_validate_schema` tests deleted alongside the tool

---

## What Is Not Changed

- `load_dataset`, `check_missing_values`, `check_data_quality`, `check_data_drift`
- `data_agent.py` builder (tool list updated, but builder structure unchanged)
- Graph topology, supervisor, all other nodes
- `AgentState` schema (no new fields needed)
