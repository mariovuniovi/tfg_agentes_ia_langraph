# Schema-Driven Data Validation Design

**Date:** 2026-04-20  
**Updated:** 2026-04-22  
**Status:** Approved

## Problem

The current data validation agent only checks that a `target` column is present and runs generic Evidently quality checks. It has no knowledge of the canonical dataset shape, cannot map raw column names to expected ones, does not support multi-file inputs, and produces no cleaned output file for downstream nodes.

## Goal

Replace the generic validation with a schema-driven approach: a committed JSON schema defines the canonical dataset (columns, types, constraints, mapping hints, join keys). The user provides one or more raw CSV files. The data validation agent loads all raw files, uses the schema to infer which file contributes which canonical columns and which column is the join key, merges the files, renames columns to canonical names, and validates all constraints. The result is a single cleaned CSV that downstream nodes consume.

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
      "name": "sample_id",
      "dtype": "int",
      "description": "Unique sample identifier used to join measurement files",
      "required": true,
      "nullable": false,
      "is_key": true,
      "mapping_hint": "Usually named 'id', 'sample_id', 'Id' or similar"
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
| `is_key` | bool | no | Marks this column as the join key used to merge raw files |
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
  ├─ reads state.dataset_paths      →  list of user-provided raw CSV paths
  ├─ builds HumanMessage:
  │    "Raw files: [<path1>, <path2>, ...]
  │     Target schema:
  │     <schema JSON>"
  │
  └─ invokes data_agent (ReAct loop)
       │
       ├─ load_dataset              [existing] inspect columns/types per file
       ├─ merge_datasets            [new] join raw files on inferred key → merged CSV
       ├─ apply_column_mapping      [new] rename/drop/reorder → canonical CSV
       ├─ validate_against_schema   [new] check all constraints → pass/fail report
       └─ check_data_quality        [existing, optional] Evidently quality report
```

After a successful run, `data_validator_node` updates `AgentState.dataset_path` to point to the canonical CSV (`data/processed/<schema_name>.csv`) so training uses the clean file.

---

## AgentState Changes

`dataset_path: str` (single input file) is replaced by `dataset_paths: list[str]` for the raw input files. The output remains a single `dataset_path` pointing to the processed CSV.

```python
# agent_state.py
dataset_paths: list[str]   # raw input files provided by the user (replaces dataset_path as input)
dataset_path: str          # cleaned output file written by data_validator_node
```

The dashboard and `run_pipeline.py` must pass `dataset_paths` instead of `dataset_path` when invoking the graph.

---

## Components

### `settings.dataset_schema` (config/settings.py)

New field with default `"iris_classification"`. No `.env` change required for the demo.

```python
dataset_schema: str = "iris_classification"
```

### `merge_datasets` (tools/data_tools.py) — NEW

```
merge_datasets(file_paths, join_spec_json, output_path) -> str
```

- `file_paths`: list of raw CSV paths
- `join_spec_json`: JSON object produced by the LLM:
  ```json
  {
    "join_key": "id",
    "files": [
      {"path": "raw/measurements.csv", "key_column": "Id"},
      {"path": "raw/labels.csv",       "key_column": "sample_id"}
    ]
  }
  ```
- Performs an inner join across all files on the specified key columns, writes to `output_path`
- Returns JSON with `{success, output_path, row_count, columns}`
- Errors gracefully (returns error JSON, not exception) if the join key is missing from any file or the join produces zero rows

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

1. Call `load_dataset` on each raw file to inspect its column names and types.
2. Use the schema (already in context), its `mapping_hint` fields, and the `is_key` marker to decide:
   - Which file contains which canonical columns.
   - Which column in each file corresponds to the join key.
3. Call `merge_datasets` with your join specification. If any file has no matching join key, report it and stop.
4. Call `apply_column_mapping` on the merged file with a mapping of merged column names to canonical names.
5. Call `validate_against_schema` on the canonical output file.
6. Optionally call `check_data_quality` for an Evidently summary.
7. Report: PASSED or FAILED, which files were merged on what keys, which columns were mapped from what, any violations.

---

## Testing

### `merge_datasets`
- Two files joined on a common key → merged CSV has columns from both
- Key column missing from one file → returns error JSON, no exception
- Join produces zero rows (no matching keys) → returns error JSON
- Output path is written correctly

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
