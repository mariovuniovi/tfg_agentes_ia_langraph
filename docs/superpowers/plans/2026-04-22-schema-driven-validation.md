# Schema-Driven Multi-File Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace single-file generic validation with schema-driven multi-file merging: user provides N raw CSVs, the agent infers which files join on which key column (guided by `is_key` in the schema), merges them, renames columns to canonical names, and validates all constraints.

**Architecture:** Three new deterministic tools (`merge_datasets`, `apply_column_mapping`, `validate_against_schema`) handle execution; the LLM agent uses schema context to produce the join spec and column mapping JSON. `AgentState` gains `dataset_paths: list[str]` as the pipeline input; `dataset_path: str` stays as the processed-output field. `data_validator_node` builds the agent's context from `dataset_paths` + schema JSON.

**Tech Stack:** pandas (merge/rename), pathlib, LangChain `@tool`, Pydantic Settings, Streamlit multiselect, pytest with real DataFrames (no LLM mocks for tools).

**Spec:** `docs/superpowers/specs/2026-04-20-schema-driven-validation-design.md`

---

### Task 1: Schema file + settings field

**Files:**
- Create: `data/schemas/iris_classification.json`
- Modify: `src/mlops_agents/config/settings.py:6-41`

- [ ] **Step 1: Create the data/schemas directory and iris schema file**

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
      "allowed_values": ["setosa", "versicolor", "virginica"],
      "mapping_hint": "Often named 'species', 'class', 'label', or 'variety' in raw datasets"
    }
  ]
}
```

Save to `data/schemas/iris_classification.json`.

- [ ] **Step 2: Add `dataset_schema` field to Settings**

In `src/mlops_agents/config/settings.py`, add after the `data_dir` field (line 38):

```python
    dataset_schema: str = "iris_classification"
```

- [ ] **Step 3: Verify settings loads without error**

Run: `uv run python -c "from mlops_agents.config.settings import settings; print(settings.dataset_schema)"`
Expected output: `iris_classification`

- [ ] **Step 4: Commit**

```bash
git add data/schemas/iris_classification.json src/mlops_agents/config/settings.py
git commit -m "feat: add iris schema file and dataset_schema settings field"
```

---

### Task 2: `validate_against_schema` tool + tests

**Files:**
- Modify: `tests/conftest.py` (add `iris_schema_file` fixture)
- Modify: `tests/test_tools/test_data_tools.py` (add tests)
- Modify: `src/mlops_agents/tools/data_tools.py` (add tool)

Note: `validate_against_schema` accepts `schema_path: str` (full path to the JSON file) rather than just a schema name. This makes the tool testable without filesystem coupling. The `data_validator_node` resolves the full path before invoking the agent.

- [ ] **Step 1: Add `iris_schema_file` fixture to conftest.py**

In `tests/conftest.py`, add after the existing fixtures:

```python
@pytest.fixture()
def iris_schema_file(tmp_path: Path) -> Path:
    """Iris schema JSON file for validate_against_schema tests."""
    import json
    schema = {
        "name": "iris_classification",
        "columns": [
            {"name": "sepal_length", "dtype": "float", "required": True, "nullable": False, "min": 0.0, "max": 30.0},
            {"name": "sepal_width",  "dtype": "float", "required": True, "nullable": False, "min": 0.0, "max": 30.0},
            {"name": "petal_length", "dtype": "float", "required": True, "nullable": False, "min": 0.0, "max": 30.0},
            {"name": "petal_width",  "dtype": "float", "required": True, "nullable": False, "min": 0.0, "max": 30.0},
            {"name": "sample_id",    "dtype": "int",   "required": True, "nullable": False, "is_key": True},
            {"name": "target",       "dtype": "str",   "required": True, "nullable": False,
             "allowed_values": ["setosa", "versicolor", "virginica"]},
        ],
    }
    path = tmp_path / "iris_classification.json"
    path.write_text(json.dumps(schema))
    return path


@pytest.fixture()
def canonical_iris_csv(tmp_path: Path) -> Path:
    """Valid canonical iris CSV — all constraints satisfied."""
    df = pd.DataFrame({
        "sepal_length": [5.1, 4.9, 4.7],
        "sepal_width":  [3.5, 3.0, 3.2],
        "petal_length": [1.4, 1.4, 1.3],
        "petal_width":  [0.2, 0.2, 0.2],
        "sample_id":    [1, 2, 3],
        "target":       ["setosa", "setosa", "setosa"],
    })
    path = tmp_path / "canonical.csv"
    df.to_csv(path, index=False)
    return path
```

- [ ] **Step 2: Write failing tests for `validate_against_schema`**

Append to `tests/test_tools/test_data_tools.py`:

```python
# ---------------------------------------------------------------------------
# validate_against_schema
# ---------------------------------------------------------------------------

def test_validate_against_schema_passes_valid_data(canonical_iris_csv, iris_schema_file):
    from mlops_agents.tools.data_tools import validate_against_schema
    result = json.loads(validate_against_schema.invoke({
        "canonical_path": str(canonical_iris_csv),
        "schema_path": str(iris_schema_file),
    }))
    assert result["passed"] is True
    assert result["violations"] == []


def test_validate_against_schema_detects_nullable_violation(tmp_path, iris_schema_file):
    from mlops_agents.tools.data_tools import validate_against_schema
    df = pd.DataFrame({
        "sepal_length": [5.1, None],
        "sepal_width":  [3.5, 3.0],
        "petal_length": [1.4, 1.4],
        "petal_width":  [0.2, 0.2],
        "sample_id":    [1, 2],
        "target":       ["setosa", "setosa"],
    })
    path = tmp_path / "nullable_violation.csv"
    df.to_csv(path, index=False)
    result = json.loads(validate_against_schema.invoke({
        "canonical_path": str(path),
        "schema_path": str(iris_schema_file),
    }))
    assert result["passed"] is False
    assert any(v["column"] == "sepal_length" and v["rule"] == "nullable" for v in result["violations"])


def test_validate_against_schema_detects_min_violation(tmp_path, iris_schema_file):
    from mlops_agents.tools.data_tools import validate_against_schema
    df = pd.DataFrame({
        "sepal_length": [-1.0, 4.9],
        "sepal_width":  [3.5, 3.0],
        "petal_length": [1.4, 1.4],
        "petal_width":  [0.2, 0.2],
        "sample_id":    [1, 2],
        "target":       ["setosa", "setosa"],
    })
    path = tmp_path / "min_violation.csv"
    df.to_csv(path, index=False)
    result = json.loads(validate_against_schema.invoke({
        "canonical_path": str(path),
        "schema_path": str(iris_schema_file),
    }))
    assert result["passed"] is False
    assert any(v["column"] == "sepal_length" and v["rule"] == "min" for v in result["violations"])


def test_validate_against_schema_detects_allowed_values_violation(tmp_path, iris_schema_file):
    from mlops_agents.tools.data_tools import validate_against_schema
    df = pd.DataFrame({
        "sepal_length": [5.1],
        "sepal_width":  [3.5],
        "petal_length": [1.4],
        "petal_width":  [0.2],
        "sample_id":    [1],
        "target":       ["unknown_species"],
    })
    path = tmp_path / "allowed_violation.csv"
    df.to_csv(path, index=False)
    result = json.loads(validate_against_schema.invoke({
        "canonical_path": str(path),
        "schema_path": str(iris_schema_file),
    }))
    assert result["passed"] is False
    assert any(v["column"] == "target" and v["rule"] == "allowed_values" for v in result["violations"])


def test_validate_against_schema_detects_missing_required_column(tmp_path, iris_schema_file):
    from mlops_agents.tools.data_tools import validate_against_schema
    df = pd.DataFrame({
        "sepal_length": [5.1],
        "sepal_width":  [3.5],
        # petal_length missing
        "petal_width":  [0.2],
        "sample_id":    [1],
        "target":       ["setosa"],
    })
    path = tmp_path / "missing_col.csv"
    df.to_csv(path, index=False)
    result = json.loads(validate_against_schema.invoke({
        "canonical_path": str(path),
        "schema_path": str(iris_schema_file),
    }))
    assert result["passed"] is False
    assert any(v["column"] == "petal_length" and v["rule"] == "required" for v in result["violations"])
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_tools/test_data_tools.py::test_validate_against_schema_passes_valid_data -v`
Expected: FAIL with `ImportError` or `AttributeError` (tool not yet defined)

- [ ] **Step 4: Implement `validate_against_schema` in data_tools.py**

Add after the `check_missing_values` tool in `src/mlops_agents/tools/data_tools.py`:

```python
@tool
def validate_against_schema(canonical_path: str, schema_path: str) -> str:
    """Validate a canonical CSV against all constraints in a schema JSON file.

    Args:
        canonical_path: Path to the cleaned/canonical CSV to validate.
        schema_path: Full path to the schema JSON file.

    Returns:
        JSON with {passed: bool, violations: [{column, rule, detail}, ...]}.
    """
    schema_file = Path(schema_path)
    if not schema_file.exists():
        return json.dumps({"error": f"Schema file not found: {schema_path}"})

    csv_file = Path(canonical_path)
    if not csv_file.exists():
        return json.dumps({"error": f"Dataset not found: {canonical_path}"})

    schema = json.loads(schema_file.read_text())
    df = pd.read_csv(csv_file)
    violations: list[dict] = []

    for col_def in schema.get("columns", []):
        name = col_def["name"]
        required = col_def.get("required", False)

        if name not in df.columns:
            if required:
                violations.append({"column": name, "rule": "required", "detail": "Column missing from dataset"})
            continue

        series = df[name]

        if not col_def.get("nullable", True) and series.isnull().any():
            null_count = int(series.isnull().sum())
            violations.append({"column": name, "rule": "nullable", "detail": f"{null_count} null value(s) found"})

        if col_def.get("unique", False) and series.duplicated().any():
            dup_count = int(series.duplicated().sum())
            violations.append({"column": name, "rule": "unique", "detail": f"{dup_count} duplicate value(s) found"})

        if "min" in col_def:
            below = series.dropna() < col_def["min"]
            if below.any():
                violations.append({"column": name, "rule": "min", "detail": f"{int(below.sum())} value(s) below minimum {col_def['min']}"})

        if "max" in col_def:
            above = series.dropna() > col_def["max"]
            if above.any():
                violations.append({"column": name, "rule": "max", "detail": f"{int(above.sum())} value(s) above maximum {col_def['max']}"})

        if "allowed_values" in col_def:
            allowed = set(col_def["allowed_values"])
            bad = series.dropna()[~series.dropna().astype(str).isin(allowed)]
            if not bad.empty:
                bad_vals = bad.unique().tolist()[:5]
                violations.append({"column": name, "rule": "allowed_values", "detail": f"Unexpected values: {bad_vals}"})

    result = {"passed": len(violations) == 0, "violations": violations}
    logger.info(f"Schema validation: {'PASSED' if result['passed'] else 'FAILED'} ({len(violations)} violation(s))")
    return json.dumps(result)
```

- [ ] **Step 5: Run all validate_against_schema tests**

Run: `uv run pytest tests/test_tools/test_data_tools.py -k "validate_against_schema" -v`
Expected: 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add tests/conftest.py tests/test_tools/test_data_tools.py src/mlops_agents/tools/data_tools.py
git commit -m "feat: add validate_against_schema tool with constraint checking"
```

---

### Task 3: `apply_column_mapping` tool + tests

**Files:**
- Modify: `tests/test_tools/test_data_tools.py` (add tests)
- Modify: `src/mlops_agents/tools/data_tools.py` (add tool)

- [ ] **Step 1: Write failing tests for `apply_column_mapping`**

Append to `tests/test_tools/test_data_tools.py`:

```python
# ---------------------------------------------------------------------------
# apply_column_mapping
# ---------------------------------------------------------------------------

def test_apply_column_mapping_renames_columns(tmp_path, sample_csv):
    from mlops_agents.tools.data_tools import apply_column_mapping
    mapping = json.dumps({"feature_1": "sepal_length", "feature_2": "sepal_width", "target": "target"})
    output_path = str(tmp_path / "canonical.csv")
    result = json.loads(apply_column_mapping.invoke({
        "raw_path": str(sample_csv),
        "mapping_json": mapping,
        "output_path": output_path,
    }))
    assert result["success"] is True
    out_df = pd.read_csv(output_path)
    assert "sepal_length" in out_df.columns
    assert "sepal_width" in out_df.columns
    assert "feature_1" not in out_df.columns


def test_apply_column_mapping_drops_unmapped_columns(tmp_path, sample_csv):
    from mlops_agents.tools.data_tools import apply_column_mapping
    # Only map feature_1 → sepal_length; feature_2 and target get dropped
    mapping = json.dumps({"feature_1": "sepal_length"})
    output_path = str(tmp_path / "canonical.csv")
    result = json.loads(apply_column_mapping.invoke({
        "raw_path": str(sample_csv),
        "mapping_json": mapping,
        "output_path": output_path,
    }))
    assert result["success"] is True
    out_df = pd.read_csv(output_path)
    assert list(out_df.columns) == ["sepal_length"]
    assert "feature_2" not in out_df.columns
    assert "dropped_columns" in result


def test_apply_column_mapping_writes_output_file(tmp_path, sample_csv):
    from mlops_agents.tools.data_tools import apply_column_mapping
    mapping = json.dumps({"feature_1": "sepal_length", "target": "target"})
    output_path = str(tmp_path / "out.csv")
    apply_column_mapping.invoke({
        "raw_path": str(sample_csv),
        "mapping_json": mapping,
        "output_path": output_path,
    })
    assert Path(output_path).exists()


def test_apply_column_mapping_reports_mapped_columns(tmp_path, sample_csv):
    from mlops_agents.tools.data_tools import apply_column_mapping
    mapping = json.dumps({"feature_1": "sepal_length", "target": "target"})
    output_path = str(tmp_path / "out.csv")
    result = json.loads(apply_column_mapping.invoke({
        "raw_path": str(sample_csv),
        "mapping_json": mapping,
        "output_path": output_path,
    }))
    assert "mapped_columns" in result
    assert "sepal_length" in result["mapped_columns"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tools/test_data_tools.py -k "apply_column_mapping" -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement `apply_column_mapping` in data_tools.py**

Add after `validate_against_schema` in `src/mlops_agents/tools/data_tools.py`:

```python
@tool
def apply_column_mapping(raw_path: str, mapping_json: str, output_path: str) -> str:
    """Rename raw columns to canonical names and write the result to a new CSV.

    Args:
        raw_path: Path to the raw (possibly merged) CSV.
        mapping_json: JSON object {"raw_col": "canonical_col", ...}.
        output_path: Destination path for the renamed CSV.

    Returns:
        JSON with {success, output_path, mapped_columns, dropped_columns}.
    """
    csv_file = Path(raw_path)
    if not csv_file.exists():
        return json.dumps({"error": f"File not found: {raw_path}"})

    mapping: dict = json.loads(mapping_json)
    df = pd.read_csv(csv_file)

    df = df.rename(columns=mapping)
    canonical_cols = list(mapping.values())
    dropped = [c for c in df.columns if c not in canonical_cols]
    df = df[canonical_cols]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    result = {
        "success": True,
        "output_path": output_path,
        "mapped_columns": canonical_cols,
        "dropped_columns": dropped,
    }
    logger.info(f"Column mapping applied: {len(canonical_cols)} mapped, {len(dropped)} dropped → {output_path}")
    return json.dumps(result)
```

- [ ] **Step 4: Run apply_column_mapping tests**

Run: `uv run pytest tests/test_tools/test_data_tools.py -k "apply_column_mapping" -v`
Expected: 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_tools/test_data_tools.py src/mlops_agents/tools/data_tools.py
git commit -m "feat: add apply_column_mapping tool"
```

---

### Task 4: `merge_datasets` tool + tests

**Files:**
- Modify: `tests/conftest.py` (add multi-file fixtures)
- Modify: `tests/test_tools/test_data_tools.py` (add tests)
- Modify: `src/mlops_agents/tools/data_tools.py` (add tool)

- [ ] **Step 1: Add multi-file fixtures to conftest.py**

Append to `tests/conftest.py`:

```python
@pytest.fixture()
def measurements_csv(tmp_path: Path) -> Path:
    """Raw iris measurements CSV with 'Id' as key column."""
    df = pd.DataFrame({
        "Id": [1, 2, 3],
        "SepalLengthCm": [5.1, 4.9, 4.7],
        "SepalWidthCm":  [3.5, 3.0, 3.2],
        "PetalLengthCm": [1.4, 1.4, 1.3],
        "PetalWidthCm":  [0.2, 0.2, 0.2],
    })
    path = tmp_path / "measurements.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def labels_csv(tmp_path: Path) -> Path:
    """Raw iris labels CSV with 'sample_id' as key column."""
    df = pd.DataFrame({
        "sample_id": [1, 2, 3],
        "species":   ["setosa", "setosa", "setosa"],
    })
    path = tmp_path / "labels.csv"
    df.to_csv(path, index=False)
    return path
```

- [ ] **Step 2: Write failing tests for `merge_datasets`**

Append to `tests/test_tools/test_data_tools.py`:

```python
# ---------------------------------------------------------------------------
# merge_datasets
# ---------------------------------------------------------------------------

def test_merge_datasets_joins_two_files(tmp_path, measurements_csv, labels_csv):
    from mlops_agents.tools.data_tools import merge_datasets
    join_spec = json.dumps({
        "join_key": "sample_id",
        "files": [
            {"path": str(measurements_csv), "key_column": "Id"},
            {"path": str(labels_csv),       "key_column": "sample_id"},
        ],
    })
    output_path = str(tmp_path / "merged.csv")
    result = json.loads(merge_datasets.invoke({
        "join_spec_json": join_spec,
        "output_path": output_path,
    }))
    assert result["success"] is True
    merged = pd.read_csv(output_path)
    assert "SepalLengthCm" in merged.columns
    assert "species" in merged.columns
    assert result["row_count"] == 3


def test_merge_datasets_writes_output_file(tmp_path, measurements_csv, labels_csv):
    from mlops_agents.tools.data_tools import merge_datasets
    join_spec = json.dumps({
        "join_key": "sample_id",
        "files": [
            {"path": str(measurements_csv), "key_column": "Id"},
            {"path": str(labels_csv),       "key_column": "sample_id"},
        ],
    })
    output_path = str(tmp_path / "merged.csv")
    merge_datasets.invoke({"join_spec_json": join_spec, "output_path": output_path})
    assert Path(output_path).exists()


def test_merge_datasets_returns_error_when_key_missing(tmp_path, measurements_csv, labels_csv):
    from mlops_agents.tools.data_tools import merge_datasets
    join_spec = json.dumps({
        "join_key": "sample_id",
        "files": [
            {"path": str(measurements_csv), "key_column": "nonexistent_key"},
            {"path": str(labels_csv),       "key_column": "sample_id"},
        ],
    })
    output_path = str(tmp_path / "merged.csv")
    result = json.loads(merge_datasets.invoke({
        "join_spec_json": join_spec,
        "output_path": output_path,
    }))
    assert "error" in result
    assert not Path(output_path).exists()


def test_merge_datasets_returns_error_when_join_produces_zero_rows(tmp_path):
    from mlops_agents.tools.data_tools import merge_datasets
    df_a = pd.DataFrame({"id": [1, 2], "val_a": [10, 20]})
    df_b = pd.DataFrame({"id": [3, 4], "val_b": ["x", "y"]})
    path_a = tmp_path / "a.csv"
    path_b = tmp_path / "b.csv"
    df_a.to_csv(path_a, index=False)
    df_b.to_csv(path_b, index=False)
    join_spec = json.dumps({
        "join_key": "id",
        "files": [
            {"path": str(path_a), "key_column": "id"},
            {"path": str(path_b), "key_column": "id"},
        ],
    })
    output_path = str(tmp_path / "merged.csv")
    result = json.loads(merge_datasets.invoke({
        "join_spec_json": join_spec,
        "output_path": output_path,
    }))
    assert "error" in result
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_tools/test_data_tools.py -k "merge_datasets" -v`
Expected: FAIL with `ImportError`

- [ ] **Step 4: Implement `merge_datasets` in data_tools.py**

Add after `apply_column_mapping` in `src/mlops_agents/tools/data_tools.py`:

```python
@tool
def merge_datasets(join_spec_json: str, output_path: str) -> str:
    """Merge multiple raw CSV files by joining them on a common key column.

    Args:
        join_spec_json: JSON with shape:
            {
              "join_key": "canonical_key_name",
              "files": [{"path": "...", "key_column": "raw_col_name"}, ...]
            }
        output_path: Destination path for the merged CSV.

    Returns:
        JSON with {success, output_path, row_count, columns} or {error}.
    """
    spec: dict = json.loads(join_spec_json)
    join_key: str = spec["join_key"]
    file_specs: list[dict] = spec["files"]

    dfs: list[pd.DataFrame] = []
    for fs in file_specs:
        path = Path(fs["path"])
        key_col = fs["key_column"]
        if not path.exists():
            return json.dumps({"error": f"File not found: {fs['path']}"})
        df = pd.read_csv(path)
        if key_col not in df.columns:
            return json.dumps({"error": f"Key column '{key_col}' not found in {fs['path']}"})
        df = df.rename(columns={key_col: join_key})
        dfs.append(df)

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on=join_key, how="inner")

    if merged.empty:
        return json.dumps({"error": "Merge produced zero rows — no matching keys across files"})

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    result = {
        "success": True,
        "output_path": output_path,
        "row_count": len(merged),
        "columns": merged.columns.tolist(),
    }
    logger.info(f"Merged {len(file_specs)} files → {len(merged)} rows, {len(merged.columns)} columns → {output_path}")
    return json.dumps(result)
```

- [ ] **Step 5: Run merge_datasets tests**

Run: `uv run pytest tests/test_tools/test_data_tools.py -k "merge_datasets" -v`
Expected: 4 tests PASS

- [ ] **Step 6: Run all data_tools tests**

Run: `uv run pytest tests/test_tools/test_data_tools.py -v`
Expected: all pass (validate_schema tests still exist and pass)

- [ ] **Step 7: Commit**

```bash
git add tests/conftest.py tests/test_tools/test_data_tools.py src/mlops_agents/tools/data_tools.py
git commit -m "feat: add merge_datasets tool with multi-file inner join"
```

---

### Task 5: Remove `validate_schema` + update data_agent builder

**Files:**
- Modify: `src/mlops_agents/tools/data_tools.py` (remove `validate_schema`)
- Modify: `src/mlops_agents/agents/data_agent.py` (swap tool list)
- Modify: `tests/test_tools/test_data_tools.py` (remove validate_schema tests, update import)

- [ ] **Step 1: Delete the `validate_schema` function from data_tools.py**

Remove the entire `validate_schema` function (lines 46-68 in the original file):

```python
# DELETE this entire function:
@tool
def validate_schema(dataset_path: str, expected_columns: str) -> str:
    ...
```

- [ ] **Step 2: Delete validate_schema tests from test_data_tools.py**

Remove the entire `# validate_schema` section (all 5 test functions: `test_validate_schema_passes_when_all_columns_present`, `test_validate_schema_passes_with_subset_of_columns`, `test_validate_schema_fails_when_column_missing`, `test_validate_schema_reports_extra_columns`, `test_validate_schema_returns_total_column_count`).

Update the import at the top of `tests/test_tools/test_data_tools.py` from:

```python
from mlops_agents.tools.data_tools import check_missing_values, load_dataset, validate_schema
```

to:

```python
from mlops_agents.tools.data_tools import (
    apply_column_mapping,
    check_missing_values,
    load_dataset,
    merge_datasets,
    validate_against_schema,
)
```

- [ ] **Step 3: Update data_agent.py tool list**

Replace the content of `src/mlops_agents/agents/data_agent.py` with:

```python
"""Data Validation Agent — validates datasets before they enter the pipeline."""

from langchain.agents import create_agent

from mlops_agents.prompts import get_prompt
from mlops_agents.tools.data_tools import (
    apply_column_mapping,
    check_missing_values,
    load_dataset,
    merge_datasets,
    validate_against_schema,
)
from mlops_agents.tools.evidently_tools import check_data_drift, check_data_quality
from mlops_agents.utils.llm import get_llm


def build_data_agent():
    """Build and return the data validation react agent."""
    return create_agent(
        model=get_llm(),
        tools=[
            load_dataset,
            merge_datasets,
            apply_column_mapping,
            validate_against_schema,
            check_missing_values,
            check_data_quality,
            check_data_drift,
        ],
        name="data_validator",
        system_prompt=get_prompt("data_agent").template,
    )
```

- [ ] **Step 4: Run all data_tools tests**

Run: `uv run pytest tests/test_tools/test_data_tools.py -v`
Expected: all tests pass (validate_schema tests gone, new tools pass)

- [ ] **Step 5: Commit**

```bash
git add src/mlops_agents/tools/data_tools.py src/mlops_agents/agents/data_agent.py tests/test_tools/test_data_tools.py
git commit -m "refactor: replace validate_schema with validate_against_schema; update data_agent tool list"
```

---

### Task 6: Update AgentState + data_validator_node

**Files:**
- Modify: `src/mlops_agents/state/agent_state.py:23`
- Modify: `src/mlops_agents/graphs/mlops_graph.py:52-67` (node) and `194-215` (main)

- [ ] **Step 1: Add `dataset_paths` to AgentState**

In `src/mlops_agents/state/agent_state.py`, replace line 23:

```python
    # Pipeline inputs
    dataset_path: str
```

with:

```python
    # Pipeline inputs
    dataset_paths: list[str]   # raw CSV files provided by user
    dataset_path: str          # canonical CSV written by data_validator_node
```

- [ ] **Step 2: Rewrite `data_validator_node` in mlops_graph.py**

Replace the `data_validator_node` function (lines 52-67) with:

```python
def data_validator_node(state: AgentState) -> Command[Literal["supervisor"]]:
    from pathlib import Path as _Path
    from mlops_agents.config.settings import settings

    schema_file = _Path("data/schemas") / f"{settings.dataset_schema}.json"
    schema_json = schema_file.read_text() if schema_file.exists() else "{}"
    schema_path = str(schema_file.resolve())

    dataset_paths = state.get("dataset_paths", [])
    context_message = HumanMessage(
        content=(
            f"Raw files: {json.dumps(dataset_paths)}\n"
            f"Schema path: {schema_path}\n"
            f"Target schema:\n{schema_json}"
        )
    )

    agent = get_agent("data_validator")
    result = agent.invoke({"messages": list(state["messages"]) + [context_message]})
    final_message = result["messages"][-1].content

    quality_report: dict = _extract_tool_json(result["messages"], "check_data_quality")
    mapping_result: dict = _extract_tool_json(result["messages"], "apply_column_mapping")
    validation_result: dict = _extract_tool_json(result["messages"], "validate_against_schema")

    processed_path = mapping_result.get("output_path", "")
    validation_passed = bool(validation_result.get("passed", False))

    logger.info("[data_validator] completed — routing back to supervisor")
    return Command(
        update={
            "messages": [HumanMessage(content=final_message, name="data_validator")],
            "validation_report": quality_report,
            "validation_passed": validation_passed,
            "dataset_path": processed_path,
        },
        goto="supervisor",
    )
```

- [ ] **Step 3: Update `main()` in mlops_graph.py to use dataset_paths**

Replace the `main()` function body in `mlops_graph.py` (lines 190-231). Change the argument parsing and initial state:

```python
def main() -> None:
    """Run the full MLOps pipeline from the CLI, including HITL approval."""
    import sys

    dataset_paths = sys.argv[1:] if len(sys.argv) > 1 else ["./data/samples/iris_measurements.csv", "./data/samples/iris_labels.csv"]
    paths_display = ", ".join(dataset_paths)

    config = {"configurable": {"thread_id": "pipeline-1"}, "recursion_limit": GRAPH_RECURSION_LIMIT}
    initial_state: dict = {
        "messages": [
            HumanMessage(content=f"Run the full MLOps pipeline on these raw files: {paths_display}")
        ],
        "next": "",
        "dataset_paths": dataset_paths,
        "dataset_path": "",
        "validation_passed": False,
        "validation_report": {},
        "trained_model_path": "",
        "training_run_id": "",
        "training_metrics": {},
        "evaluation_passed": False,
        "evaluation_report": {},
        "best_model_uri": "",
        "deployment_decision": "pending",
        "deployment_status": "",
        "error_message": "",
        "retry_count": 0,
    }

    print(f"\n{'='*60}")
    print(f"MLOps Pipeline — files: {paths_display}")
    print(f"{'='*60}\n")

    for event in graph.stream(initial_state, config=config):
        if "__interrupt__" in event:
            interrupt_value = event["__interrupt__"][0].value
            _handle_hitl(graph, config, interrupt_value)
        else:
            for node_name in event:
                print(f"  [{node_name}] completed")

    print(f"\n{'='*60}")
    print("Pipeline finished.")
    print(f"{'='*60}\n")
```

- [ ] **Step 4: Run unit tests to verify no regressions**

Run: `uv run pytest -m "not integration" -v`
Expected: all existing unit tests pass

- [ ] **Step 5: Commit**

```bash
git add src/mlops_agents/state/agent_state.py src/mlops_agents/graphs/mlops_graph.py
git commit -m "feat: update AgentState and data_validator_node for multi-file pipeline input"
```

---

### Task 7: Update pipeline_helpers + dashboard

**Files:**
- Modify: `dashboard/pipeline_helpers.py:41-59`
- Modify: `dashboard/pages/01_pipeline.py:162-225`

- [ ] **Step 1: Update `build_initial_state` in pipeline_helpers.py**

Replace the `build_initial_state` function (lines 41-59):

```python
def build_initial_state(dataset_paths: list[str]) -> dict:
    """Build the initial LangGraph state dict for a pipeline run."""
    paths_display = ", ".join(dataset_paths)
    return {
        "messages": [HumanMessage(content=f"Run the full MLOps pipeline on these raw files: {paths_display}")],
        "next": "",
        "dataset_paths": dataset_paths,
        "dataset_path": "",
        "validation_passed": False,
        "validation_report": {},
        "trained_model_path": "",
        "training_run_id": "",
        "training_metrics": {},
        "evaluation_passed": False,
        "evaluation_report": {},
        "best_model_uri": "",
        "deployment_decision": "pending",
        "deployment_status": "",
        "error_message": "",
        "retry_count": 0,
    }
```

- [ ] **Step 2: Update the idle phase in 01_pipeline.py to multiselect**

Replace the idle-phase dataset selector block (lines 162-225). Change the `st.selectbox` to `st.multiselect` and pass `dataset_paths` instead of `dataset_path`:

```python
if st.session_state["phase"] == "idle":
    from mlops_agents.config.settings import settings

    data_dir = Path(settings.data_dir)
    csvs = sorted(data_dir.glob("*.csv")) if data_dir.exists() else []
    options = [str(f) for f in csvs] or ["./data/samples/iris_measurements.csv", "./data/samples/iris_labels.csv"]

    col1, col2 = st.columns([3, 1])
    with col1:
        dataset_paths = st.multiselect(
            "Select raw dataset files (one or more CSVs to merge)",
            options=options,
            default=options[:2] if len(options) >= 2 else options,
            help="Select all CSV files that together form the target dataset",
        )
    with col2:
        run_button = st.button("▶ Run Pipeline", type="primary", use_container_width=True)

    if run_button:
        if not dataset_paths:
            st.error("Select at least one dataset file.")
            st.stop()

        thread_id = f"streamlit-{int(time.time())}"
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": GRAPH_RECURSION_LIMIT,
        }
        st.session_state["pipeline_config"] = config

        left_col, right_col = st.columns([4, 6])
        with left_col:
            st.subheader("Pipeline Log")
            log_placeholder = st.empty()
        with right_col:
            st.subheader("Live Results")
            right_placeholder = st.empty()

        interrupt_detected = False
        for event in graph.stream(build_initial_state(dataset_paths), config=config):
            if "__interrupt__" in event:
                st.session_state["interrupt_value"] = event["__interrupt__"][0].value
                st.session_state["phase"] = "awaiting_approval"
                _log("⏸ **Pipeline paused — awaiting human approval**")
                interrupt_detected = True
                break
            else:
                line = event_to_log_line(event)
                if line:
                    _log(line)
                    _render_log(log_placeholder)
                node = next(iter(event), None)
                if node in ("data_validator", "trainer", "evaluator"):
                    _update_panel_data(config)
                    _render_tabs(right_placeholder)

        if interrupt_detected:
            _render_log(log_placeholder)
            st.rerun()
        elif st.session_state["phase"] == "idle":
            _update_panel_data(config)
            final = graph.get_state(config).values
            st.session_state["deployment_decision"] = final.get("deployment_decision", "pending")
            msgs = final.get("messages", [])
            if msgs:
                last = msgs[-1]
                st.session_state["final_message"] = last.content if hasattr(last, "content") else str(last)
            st.session_state["phase"] = "complete"
            st.rerun()
```

- [ ] **Step 3: Run unit tests**

Run: `uv run pytest -m "not integration" -v`
Expected: all pass

- [ ] **Step 4: Commit**

```bash
git add dashboard/pipeline_helpers.py dashboard/pages/01_pipeline.py
git commit -m "feat: update dashboard and pipeline_helpers for multi-file dataset_paths input"
```

---

### Task 8: Update data_agent.yaml prompt

**Files:**
- Modify: `src/mlops_agents/prompts/data_agent.yaml`

- [ ] **Step 1: Rewrite the prompt**

Replace the entire content of `src/mlops_agents/prompts/data_agent.yaml` with:

```yaml
_type: "prompt"
input_variables: []
template: |
  You are a Data Validation Specialist responsible for gating the MLOps pipeline.

  Your context message contains:
  - "Raw files": a JSON list of raw CSV file paths provided by the user
  - "Schema path": the full path to the target schema JSON file
  - "Target schema": the full schema JSON defining canonical columns, types, constraints, and mapping hints

  Your job is to merge the raw files into the canonical dataset, validate all constraints, and report the result clearly.

  TOOLS:
  - load_dataset: Load and summarise a single CSV — use this on each raw file first.
  - merge_datasets: Join multiple CSVs on a common key column.
  - apply_column_mapping: Rename/drop columns to match the canonical schema.
  - validate_against_schema: Check all schema constraints (nullability, min/max, allowed values, required columns).
  - check_missing_values: Compute missing value statistics per column.
  - check_data_quality: Run an Evidently AI quality report.
  - check_data_drift: Compare two CSVs for statistical drift (optional, only if a reference dataset is available).

  PROCESS:
  1. Call load_dataset on EACH raw file to inspect its column names and types.
  2. Read the schema. Use the "mapping_hint" and "is_key": true fields to decide:
     - Which file contains which canonical columns.
     - Which column in each file corresponds to the join key.
  3. Call merge_datasets with your join specification. If any file lacks a matching key column, stop and report the error.
  4. Call apply_column_mapping on the merged file:
     - Build a mapping {"raw_or_merged_col": "canonical_col"} for every column you can match.
     - Use "data/processed/<schema_name>.csv" as the output path.
  5. Call validate_against_schema on the canonical output file, passing the schema path from your context.
  6. Optionally call check_data_quality for an Evidently summary.
  7. Report clearly:
     - PASSED or FAILED
     - Which files were merged on which key columns
     - Which raw columns were mapped to which canonical names
     - Any constraint violations with detail

  Be specific. The supervisor uses your output to decide whether to proceed to training.
  If validation fails, clearly explain what the data engineer needs to fix.
```

- [ ] **Step 2: Run full unit test suite**

Run: `uv run pytest -m "not integration" -v`
Expected: all tests pass

- [ ] **Step 3: Commit**

```bash
git add src/mlops_agents/prompts/data_agent.yaml
git commit -m "feat: rewrite data_agent prompt for multi-file merge and schema-driven validation workflow"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| `is_key` field in schema | Task 1 (schema file) |
| `dataset_paths: list[str]` in AgentState | Task 6 |
| `merge_datasets` tool with join spec | Task 4 |
| `apply_column_mapping` tool | Task 3 |
| `validate_against_schema` tool | Task 2 |
| Remove `validate_schema` | Task 5 |
| Update `data_agent.py` tool list | Task 5 |
| Update `data_agent.yaml` workflow | Task 8 |
| `data_validator_node` reads `dataset_paths` + schema | Task 6 |
| `dataset_path` updated to processed CSV after validation | Task 6 |
| Dashboard passes `dataset_paths` | Task 7 |
| `settings.dataset_schema` field | Task 1 |
| merge produces zero rows → error JSON | Task 4 |
| key missing from file → error JSON | Task 4 |
| All constraint types tested | Task 2 |

**Type consistency check:** `dataset_paths: list[str]` defined in Task 6 AgentState, used in Task 6 node and Task 7 helpers. `validate_against_schema` uses `schema_path: str` throughout Tasks 2, 5, 6 consistently.

**No placeholders found.**
