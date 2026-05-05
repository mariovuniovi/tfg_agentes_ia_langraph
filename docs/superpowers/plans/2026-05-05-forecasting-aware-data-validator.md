# Forecasting-Aware Data Validator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the data validator with role-aware imputation, two new temporal tools, and an updated agent prompt so forecasting datasets are handled correctly.

**Architecture:** `impute_missing_values` is restructured into a role-aware dispatcher. A private helper `_interpolate_short_internal_gaps` implements proper large-gap preservation (only short, internally-bounded gaps are filled). `_tabular_impute` protects the target column by dropping rows with missing labels. Two new `@tool` functions — `parse_datetime_column` and `detect_temporal_gaps` — are added to `data_tools.py` and registered in `data_agent.py`. `data_agent.yaml` gains a forecasting branch with an explicit 5-step tool call order.

**Tech Stack:** Python 3.12, pandas 2.x, LangChain `@tool`, pytest

---

## File Map

| File | Action | What changes |
|---|---|---|
| `src/mlops_agents/tools/data_tools.py` | Modify | Rewrite `impute_missing_values`; add `_tabular_impute`, `_forecasting_impute`, `_interpolate_short_internal_gaps` helpers; add `parse_datetime_column`, `detect_temporal_gaps` tools |
| `src/mlops_agents/agents/data_agent.py` | Modify | Import and register the two new tools |
| `src/mlops_agents/prompts/data_agent.yaml` | Modify | Add forecasting branch with explicit tool call order |
| `tests/test_tools/test_data_tools.py` | Modify | Replace old impute tests with new-signature tests; add tests for new tools |

---

## Task 1: Rewrite `impute_missing_values` into role-aware dispatcher

**Files:**
- Modify: `src/mlops_agents/tools/data_tools.py`
- Modify: `tests/test_tools/test_data_tools.py`

### Background

The current `impute_missing_values(path: str)` applies a single strategy to all columns. Key design decisions for the new version:

- **Tabular (classification/regression):** target rows with missing labels are **dropped** (not imputed — missing labels are invalid for supervised learning). All other columns get mean (numeric) or mode (categorical).
- **Forecasting target:** a private helper `_interpolate_short_internal_gaps` interpolates ONLY gaps that are (a) ≤ `max_interpolation_gap` periods long AND (b) bounded by known values on both sides. Gaps that exceed the threshold OR sit at series boundaries are left entirely unchanged. This avoids the partial-fill bug from `pandas limit=N` which fills the first N values of a large gap.
- **Forecasting exogenous:** dtype-aware — numeric gets `ffill + interpolate + bfill`; categorical gets `ffill + bfill + mode fallback`.

The old tests use `{"path": ...}` and the old return format — they are entirely replaced.

- [ ] **Step 1: Write failing tests for new `impute_missing_values` signature**

Replace the entire `# impute_missing_values` section in `tests/test_tools/test_data_tools.py` (lines 316–449) with:

```python
# ---------------------------------------------------------------------------
# impute_missing_values — role-aware dispatcher
# ---------------------------------------------------------------------------

import pytest


def test_impute_invalid_problem_type_raises(tmp_path):
    df = pd.DataFrame({"val": [1.0, None], "target": ["a", "b"]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="problem_type"):
        impute_missing_values.invoke({
            "dataset_path": str(path),
            "problem_type": "unknown",
            "target_column": "target",
        })


def test_impute_tabular_numeric_uses_mean(tmp_path):
    df = pd.DataFrame({"val": [1.0, None, 3.0], "target": ["a", "b", "c"]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    result = json.loads(impute_missing_values.invoke({
        "dataset_path": str(path),
        "problem_type": "classification",
        "target_column": "target",
    }))
    assert "val" in result["columns_imputed"]
    assert result["rows_affected"] == 1
    df_after = pd.read_csv(path)
    assert df_after["val"].isnull().sum() == 0
    assert abs(df_after["val"].iloc[1] - 2.0) < 0.01


def test_impute_tabular_categorical_uses_mode(tmp_path):
    df = pd.DataFrame({"species": ["setosa", "setosa", None], "target": [0, 1, 0]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    result = json.loads(impute_missing_values.invoke({
        "dataset_path": str(path),
        "problem_type": "regression",
        "target_column": "target",
    }))
    assert "species" in result["columns_imputed"]
    df_after = pd.read_csv(path)
    assert df_after["species"].iloc[2] == "setosa"


def test_impute_tabular_does_not_impute_target(tmp_path):
    # target column should NEVER be imputed — rows with missing target are dropped
    df = pd.DataFrame({"val": [1.0, 2.0, 3.0], "target": ["a", None, "c"]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    result = json.loads(impute_missing_values.invoke({
        "dataset_path": str(path),
        "problem_type": "classification",
        "target_column": "target",
    }))
    assert "target" not in result["columns_imputed"]
    df_after = pd.read_csv(path)
    assert df_after["target"].isnull().sum() == 0  # row was dropped, not filled
    assert len(df_after) == 2  # one row dropped


def test_impute_tabular_missing_target_reported_in_warnings(tmp_path):
    df = pd.DataFrame({"val": [1.0, 2.0], "target": [None, "b"]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    result = json.loads(impute_missing_values.invoke({
        "dataset_path": str(path),
        "problem_type": "classification",
        "target_column": "target",
    }))
    assert len(result["warnings"]) > 0
    assert "Dropped" in result["warnings"][0]


def test_impute_tabular_no_missing_is_noop(tmp_path):
    df = pd.DataFrame({"val": [1.0, 2.0], "target": [0, 1]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    result = json.loads(impute_missing_values.invoke({
        "dataset_path": str(path),
        "problem_type": "classification",
        "target_column": "target",
    }))
    assert result["columns_imputed"] == []
    assert result["rows_affected"] == 0


def test_impute_tabular_returns_output_path(tmp_path):
    df = pd.DataFrame({"val": [1.0, None], "target": [0, 1]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    result = json.loads(impute_missing_values.invoke({
        "dataset_path": str(path),
        "problem_type": "classification",
        "target_column": "target",
    }))
    assert "output_path" in result


def test_impute_tabular_writes_to_output_path(tmp_path):
    df = pd.DataFrame({"val": [1.0, None], "target": [0, 1]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    out = tmp_path / "out.csv"
    impute_missing_values.invoke({
        "dataset_path": str(path),
        "problem_type": "classification",
        "target_column": "target",
        "output_path": str(out),
    })
    assert out.exists()


def test_impute_forecasting_raises_if_datetime_null(tmp_path):
    df = pd.DataFrame({
        "date": [None, "2024-01-02", "2024-01-03"],
        "target": [1.0, 2.0, 3.0],
        "sid": ["s1", "s1", "s1"],
    })
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="null values"):
        impute_missing_values.invoke({
            "dataset_path": str(path),
            "problem_type": "forecasting",
            "target_column": "target",
            "datetime_column": "date",
            "series_id_columns": ["sid"],
        })


def test_impute_forecasting_raises_if_series_id_null(tmp_path):
    df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "target": [1.0, 2.0, 3.0],
        "sid": [None, "s1", "s1"],
    })
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="null values"):
        impute_missing_values.invoke({
            "dataset_path": str(path),
            "problem_type": "forecasting",
            "target_column": "target",
            "datetime_column": "date",
            "series_id_columns": ["sid"],
        })


def test_impute_forecasting_short_gap_target_filled(tmp_path):
    # Internal gap of 2 ≤ max_interpolation_gap=3 → all filled
    df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        "target": [1.0, None, None, 4.0, 5.0],
        "sid": ["s1"] * 5,
    })
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    result = json.loads(impute_missing_values.invoke({
        "dataset_path": str(path),
        "problem_type": "forecasting",
        "target_column": "target",
        "datetime_column": "date",
        "series_id_columns": ["sid"],
        "max_interpolation_gap": 3,
    }))
    assert result["target_large_gaps"] == []
    df_after = pd.read_csv(path)
    assert df_after["target"].isnull().sum() == 0


def test_impute_forecasting_large_gap_fully_preserved(tmp_path):
    # Internal gap of 5 > max_interpolation_gap=3 → NONE of the 5 NaNs are filled
    df = pd.DataFrame({
        "date": ["2024-01-0" + str(i) for i in range(1, 9)],
        "target": [1.0, None, None, None, None, None, 7.0, 8.0],
        "sid": ["s1"] * 8,
    })
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    result = json.loads(impute_missing_values.invoke({
        "dataset_path": str(path),
        "problem_type": "forecasting",
        "target_column": "target",
        "datetime_column": "date",
        "series_id_columns": ["sid"],
        "max_interpolation_gap": 3,
    }))
    assert len(result["target_large_gaps"]) > 0
    assert result["target_large_gaps"][0]["gap_sizes"] == [5]
    df_after = pd.read_csv(path)
    # All 5 NaNs must remain — no partial imputation of large gaps
    assert df_after["target"].isnull().sum() == 5


def test_impute_forecasting_exogenous_numeric_forward_filled(tmp_path):
    df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "target": [1.0, 2.0, 3.0],
        "exog": [10.0, None, 30.0],
        "sid": ["s1"] * 3,
    })
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    result = json.loads(impute_missing_values.invoke({
        "dataset_path": str(path),
        "problem_type": "forecasting",
        "target_column": "target",
        "datetime_column": "date",
        "series_id_columns": ["sid"],
    }))
    assert "exog" in result["columns_imputed"]
    df_after = pd.read_csv(path)
    assert df_after["exog"].isnull().sum() == 0


def test_impute_forecasting_exogenous_categorical_filled(tmp_path):
    df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "target": [1.0, 2.0, 3.0],
        "category": ["A", None, "A"],
        "sid": ["s1"] * 3,
    })
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    result = json.loads(impute_missing_values.invoke({
        "dataset_path": str(path),
        "problem_type": "forecasting",
        "target_column": "target",
        "datetime_column": "date",
        "series_id_columns": ["sid"],
    }))
    assert "category" in result["columns_imputed"]
    df_after = pd.read_csv(path)
    assert df_after["category"].isnull().sum() == 0


def test_impute_file_not_found_returns_error():
    result = json.loads(impute_missing_values.invoke({
        "dataset_path": "/nonexistent/file.csv",
        "problem_type": "classification",
        "target_column": "target",
    }))
    assert "error" in result
```

- [ ] **Step 2: Run new tests to verify they fail**

```
uv run pytest tests/test_tools/test_data_tools.py -k "impute" -v
```

Expected: new tests FAIL (wrong signature/import), old impute tests may still pass temporarily

- [ ] **Step 3: Write the three private helpers and rewrite `impute_missing_values`**

Replace the existing `impute_missing_values` function (lines 229–284) in `src/mlops_agents/tools/data_tools.py` with the following (helpers first, then the `@tool`):

```python
def _interpolate_short_internal_gaps(s: pd.Series, max_gap: int) -> tuple[pd.Series, list[int]]:
    """Interpolate only short internal gaps; leave large or boundary gaps fully unchanged.

    A gap is interpolated only when:
      (1) its length is <= max_gap, AND
      (2) it is bounded by known (non-null) values on both sides.
    Leading/trailing gaps and any gap longer than max_gap are left as-is.

    Returns the imputed series and a list of gap sizes that were NOT filled.
    """
    missing = s.isna()
    if not missing.any():
        return s.copy(), []

    interpolated = s.interpolate(method="linear", limit_area="inside")
    result = s.copy()
    large_gaps: list[int] = []

    group_id = (missing != missing.shift()).cumsum()

    for _, idx in s[missing].groupby(group_id[missing]).groups.items():
        idx = list(idx)
        gap_size = len(idx)
        positions = [s.index.get_loc(i) for i in idx]
        min_pos, max_pos = min(positions), max(positions)
        has_before = min_pos > 0 and pd.notna(s.iloc[min_pos - 1])
        has_after = max_pos < len(s) - 1 and pd.notna(s.iloc[max_pos + 1])

        if gap_size <= max_gap and has_before and has_after:
            result.loc[idx] = interpolated.loc[idx]
        else:
            large_gaps.append(gap_size)

    return result, large_gaps


def _tabular_impute(df: pd.DataFrame, target_column: str) -> dict:
    """Mean/mode imputation for classification and regression datasets.

    Target column rows with missing values are dropped (not imputed).
    All other columns: mean for numeric, mode for categorical.
    """
    warnings_list: list[str] = []

    if target_column in df.columns:
        missing_target = int(df[target_column].isnull().sum())
        if missing_target > 0:
            df = df.dropna(subset=[target_column])
            warnings_list.append(
                f"Dropped {missing_target} row(s) with missing target '{target_column}'."
            )

    imputed_cols: list[str] = []
    rows_affected = 0

    for col in df.columns:
        if col == target_column:
            continue
        null_count = int(df[col].isnull().sum())
        if null_count == 0:
            continue
        if df[col].dtype in ("float64", "int64"):
            df[col] = df[col].fillna(float(df[col].mean()))
        else:
            mode = df[col].mode()
            df[col] = df[col].fillna(str(mode.iloc[0]) if not mode.empty else "unknown")
        imputed_cols.append(col)
        rows_affected += null_count

    return {
        "df": df,
        "columns_imputed": imputed_cols,
        "rows_affected": rows_affected,
        "warnings": warnings_list,
    }


def _forecasting_impute(
    df: pd.DataFrame,
    target_column: str,
    datetime_column: str,
    series_id_columns: list[str],
    max_interpolation_gap: int,
) -> dict:
    """Time-aware imputation for forecasting datasets."""
    if df[datetime_column].isnull().any():
        raise ValueError(
            f"datetime_column '{datetime_column}' contains null values — cannot impute"
        )
    for col in series_id_columns:
        if df[col].isnull().any():
            raise ValueError(
                f"series_id_column '{col}' contains null values — cannot impute"
            )

    protected = {datetime_column} | set(series_id_columns)
    imputed_cols: set[str] = set()
    rows_affected = 0
    target_large_gaps: list[dict] = []

    def _process(grp: pd.DataFrame, series_id: dict) -> pd.DataFrame:
        nonlocal rows_affected
        g = grp.copy()
        for col in g.columns:
            if col in protected:
                continue
            null_count = int(g[col].isnull().sum())
            if null_count == 0:
                continue
            if col == target_column:
                new_series, large = _interpolate_short_internal_gaps(
                    g[col], max_interpolation_gap
                )
                filled = null_count - int(new_series.isna().sum())
                g[col] = new_series
                if filled > 0:
                    imputed_cols.add(col)
                    rows_affected += filled
                if large:
                    target_large_gaps.append({"series_id": series_id, "gap_sizes": large})
            else:
                if pd.api.types.is_numeric_dtype(g[col]):
                    g[col] = g[col].ffill().interpolate(method="linear").bfill()
                else:
                    g[col] = g[col].ffill().bfill()
                    if g[col].isnull().any():
                        mode = g[col].mode()
                        g[col] = g[col].fillna(mode.iloc[0] if not mode.empty else "unknown")
                imputed_cols.add(col)
                rows_affected += null_count
        return g

    if series_id_columns:
        parts = []
        for keys, grp in df.groupby(series_id_columns, group_keys=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            parts.append(_process(grp, dict(zip(series_id_columns, keys))))
        df = pd.concat(parts)
    else:
        df = _process(df, {})

    return {
        "df": df,
        "columns_imputed": list(imputed_cols),
        "rows_affected": rows_affected,
        "target_large_gaps": target_large_gaps,
    }


@tool
def impute_missing_values(
    dataset_path: str,
    problem_type: str,
    target_column: str,
    datetime_column: str | None = None,
    series_id_columns: list[str] | None = None,
    max_interpolation_gap: int = 3,
    output_path: str = "",
) -> str:
    """Impute missing values using a strategy appropriate for the problem type.

    For classification/regression: rows with missing target are dropped; mean for
    numeric non-target columns, mode for categoricals.
    For forecasting: protected datetime/series_id columns (raise if null); only short
    internal gaps in the target are interpolated (≤ max_interpolation_gap, bounded on
    both sides); exogenous features get dtype-aware forward-fill/interpolation.

    Args:
        dataset_path: Path to the CSV file to impute.
        problem_type: One of "classification", "regression", "forecasting".
        target_column: Name of the target/label column.
        datetime_column: Required for forecasting — the datetime index column.
        series_id_columns: Required for forecasting — columns that identify each series.
        max_interpolation_gap: Max consecutive missing periods to interpolate in target.
        output_path: Destination path for imputed CSV. Defaults to overwrite input.

    Returns:
        JSON with {output_path, columns_imputed, rows_affected, warnings} plus
        target_large_gaps for forecasting.
    """
    valid = {"classification", "regression", "forecasting"}
    if problem_type not in valid:
        raise ValueError(
            f"problem_type must be one of {sorted(valid)}, got {problem_type!r}"
        )

    path = Path(dataset_path)
    if not path.exists():
        return json.dumps({"error": f"File not found: {dataset_path}"})

    df = pd.read_csv(path)
    cols = list(series_id_columns or [])
    dest = Path(output_path) if output_path else path
    dest.parent.mkdir(parents=True, exist_ok=True)

    if problem_type in ("classification", "regression"):
        info = _tabular_impute(df, target_column)
    else:
        if datetime_column is None:
            raise ValueError("datetime_column is required for forecasting imputation")
        info = _forecasting_impute(df, target_column, datetime_column, cols, max_interpolation_gap)

    result_df: pd.DataFrame = info.pop("df")
    result_df.to_csv(dest, index=False)
    info["output_path"] = str(dest)
    logger.info(f"Imputed {len(info['columns_imputed'])} column(s) in {dest.name}")
    return json.dumps(info, default=str)
```

- [ ] **Step 4: Run new tests to verify they pass**

```
uv run pytest tests/test_tools/test_data_tools.py -k "impute" -v
```

Expected: all new impute tests PASS

- [ ] **Step 5: Verify the rest of the test suite still passes**

```
uv run pytest tests/test_tools/test_data_tools.py -v
```

Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/mlops_agents/tools/data_tools.py tests/test_tools/test_data_tools.py
git commit -m "feat: rewrite impute_missing_values as role-aware dispatcher (tabular + forecasting)"
```

---

## Task 2: Add `parse_datetime_column` tool

**Files:**
- Modify: `src/mlops_agents/tools/data_tools.py`
- Modify: `tests/test_tools/test_data_tools.py`

### Background

`parse_datetime_column` parses a string column as datetime using `errors="coerce"` (catches both original nulls and unparseable strings with one consistent error), sorts by `series_id_cols + [datetime_col]` (critical for multi-series panel data), and writes the result. This guarantees the dataset is in the right order before gap detection and imputation.

- [ ] **Step 1: Write failing tests**

Add this section to `tests/test_tools/test_data_tools.py`:

```python
# ---------------------------------------------------------------------------
# parse_datetime_column
# ---------------------------------------------------------------------------

from mlops_agents.tools.data_tools import parse_datetime_column


def test_parse_datetime_column_parses_and_sorts(tmp_path):
    df = pd.DataFrame({
        "date": ["2024-01-03", "2024-01-01", "2024-01-02"],
        "val": [3, 1, 2],
    })
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    result = json.loads(parse_datetime_column.invoke({
        "dataset_path": str(path),
        "datetime_col": "date",
    }))
    assert result["null_count"] == 0
    assert result["dtype"] == "datetime64"
    df_after = pd.read_csv(path, parse_dates=["date"])
    assert list(df_after["val"]) == [1, 2, 3]


def test_parse_datetime_column_sorts_by_series_then_date(tmp_path):
    # Multi-series: rows must be grouped by series_id, then sorted by date within each
    df = pd.DataFrame({
        "date": ["2024-01-02", "2024-01-01", "2024-01-02", "2024-01-01"],
        "store_id": ["B", "A", "A", "B"],
        "sales": [20, 10, 30, 40],
    })
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    parse_datetime_column.invoke({
        "dataset_path": str(path),
        "datetime_col": "date",
        "series_id_cols": ["store_id"],
    })
    df_after = pd.read_csv(path)
    # Should be sorted: A 2024-01-01, A 2024-01-02, B 2024-01-01, B 2024-01-02
    assert list(df_after["store_id"]) == ["A", "A", "B", "B"]
    assert list(df_after["sales"]) == [10, 30, 40, 20]


def test_parse_datetime_column_raises_if_nulls(tmp_path):
    df = pd.DataFrame({"date": [None, "2024-01-02"], "val": [1, 2]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="null"):
        parse_datetime_column.invoke({
            "dataset_path": str(path),
            "datetime_col": "date",
        })


def test_parse_datetime_column_raises_if_unparseable(tmp_path):
    # errors="coerce" converts unparseable to NaT, then null check triggers
    df = pd.DataFrame({"date": ["not-a-date", "2024-01-02"], "val": [1, 2]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="null|unparseable"):
        parse_datetime_column.invoke({
            "dataset_path": str(path),
            "datetime_col": "date",
        })


def test_parse_datetime_column_writes_to_output_path(tmp_path):
    df = pd.DataFrame({"date": ["2024-01-01", "2024-01-02"], "val": [1, 2]})
    path = tmp_path / "data.csv"
    out = tmp_path / "sorted.csv"
    df.to_csv(path, index=False)
    parse_datetime_column.invoke({
        "dataset_path": str(path),
        "datetime_col": "date",
        "output_path": str(out),
    })
    assert out.exists()


def test_parse_datetime_column_returns_error_for_missing_file():
    result = json.loads(parse_datetime_column.invoke({
        "dataset_path": "/nonexistent/data.csv",
        "datetime_col": "date",
    }))
    assert "error" in result
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_tools/test_data_tools.py -k "parse_datetime" -v
```

Expected: FAIL with ImportError or NameError

- [ ] **Step 3: Implement `parse_datetime_column`**

Add to `src/mlops_agents/tools/data_tools.py` (before `impute_missing_values`, i.e. before the `_interpolate_short_internal_gaps` helper):

```python
@tool
def parse_datetime_column(
    dataset_path: str,
    datetime_col: str,
    series_id_cols: list[str] | None = None,
    output_path: str = "",
) -> str:
    """Parse a string column as datetime, sort the dataset, and write the result.

    Uses errors="coerce" so both original nulls and unparseable strings become NaT
    and are caught by a single null check with a clear error message.

    Sorts by series_id_cols + [datetime_col] so multi-series panel data is grouped
    correctly before gap detection and position-based interpolation.

    Args:
        dataset_path: Path to the CSV file.
        datetime_col: Name of the column to parse as datetime.
        series_id_cols: Optional series identifier columns — sort by these first.
        output_path: Destination path for sorted CSV. Defaults to overwrite input.

    Returns:
        JSON with {output_path, dtype, null_count} or {error}.
    """
    path = Path(dataset_path)
    if not path.exists():
        return json.dumps({"error": f"File not found: {dataset_path}"})

    df = pd.read_csv(path)

    if datetime_col not in df.columns:
        return json.dumps({"error": f"Column '{datetime_col}' not found in dataset"})

    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")

    null_count = int(df[datetime_col].isnull().sum())
    if null_count > 0:
        raise ValueError(
            f"datetime_column '{datetime_col}' has {null_count} null or unparseable "
            "value(s) — temporal forecasting requires a complete, parseable time index"
        )

    sort_cols = list(series_id_cols or []) + [datetime_col]
    df = df.sort_values(by=sort_cols).reset_index(drop=True)

    dest = Path(output_path) if output_path else path
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)

    logger.info(f"Parsed and sorted '{datetime_col}' in {dest.name}")
    return json.dumps({"output_path": str(dest), "dtype": "datetime64", "null_count": 0})
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/test_tools/test_data_tools.py -k "parse_datetime" -v
```

Expected: all 6 tests PASS

- [ ] **Step 5: Run full test file to check for regressions**

```
uv run pytest tests/test_tools/test_data_tools.py -v
```

Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/mlops_agents/tools/data_tools.py tests/test_tools/test_data_tools.py
git commit -m "feat: add parse_datetime_column tool — coerce parse, multi-series sort"
```

---

## Task 3: Add `detect_temporal_gaps` tool

**Files:**
- Modify: `src/mlops_agents/tools/data_tools.py`
- Modify: `tests/test_tools/test_data_tools.py`

### Background

`detect_temporal_gaps` validates critical forecasting keys first (raises if datetime column doesn't exist or has nulls, any series_id column has nulls, target column is missing, frequency alias is invalid, or duplicate keys exist), then detects missing time periods per series and returns a compact summary. The full report is written to a stem-based artifact path to avoid overwriting between pipeline runs.

Key implementation details:
- Read CSV without `parse_dates` first, check column existence, then coerce with `pd.to_datetime(errors="coerce")`
- Validate `frequency` with `to_offset` before running gap detection
- Default artifact path: `artifacts/temporal_gaps_{path.stem}.json` (not a fixed global path)

- [ ] **Step 1: Write failing tests**

Add this section to `tests/test_tools/test_data_tools.py`:

```python
# ---------------------------------------------------------------------------
# detect_temporal_gaps
# ---------------------------------------------------------------------------

from mlops_agents.tools.data_tools import detect_temporal_gaps


@pytest.fixture()
def daily_series_csv(tmp_path: Path) -> Path:
    """Daily time series with one gap on 2024-01-03."""
    df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05"],
        "sales": [10.0, 20.0, 40.0, 50.0],
        "store_id": ["S01"] * 4,
    })
    path = tmp_path / "series.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture()
def daily_series_no_gaps_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "sales": [10.0, 20.0, 30.0],
        "store_id": ["S01"] * 3,
    })
    path = tmp_path / "series_ok.csv"
    df.to_csv(path, index=False)
    return path


def test_detect_temporal_gaps_finds_gap(tmp_path, daily_series_csv):
    artifact = tmp_path / "gaps.json"
    result = json.loads(detect_temporal_gaps.invoke({
        "dataset_path": str(daily_series_csv),
        "datetime_col": "date",
        "series_id_cols": ["store_id"],
        "frequency": "D",
        "target_column": "sales",
        "output_path": str(artifact),
    }))
    assert result["has_gaps"] is True
    assert result["total_missing_periods"] == 1
    assert result["n_series_with_gaps"] == 1
    assert len(result["gap_examples"]) == 1
    assert result["gap_examples"][0]["first_missing"] == "2024-01-03"


def test_detect_temporal_gaps_no_gaps(tmp_path, daily_series_no_gaps_csv):
    artifact = tmp_path / "gaps.json"
    result = json.loads(detect_temporal_gaps.invoke({
        "dataset_path": str(daily_series_no_gaps_csv),
        "datetime_col": "date",
        "series_id_cols": ["store_id"],
        "frequency": "D",
        "target_column": "sales",
        "output_path": str(artifact),
    }))
    assert result["has_gaps"] is False
    assert result["total_missing_periods"] == 0
    assert result["gap_examples"] == []


def test_detect_temporal_gaps_compact_format(tmp_path, daily_series_csv):
    artifact = tmp_path / "gaps.json"
    result = json.loads(detect_temporal_gaps.invoke({
        "dataset_path": str(daily_series_csv),
        "datetime_col": "date",
        "series_id_cols": ["store_id"],
        "frequency": "D",
        "target_column": "sales",
        "output_path": str(artifact),
    }))
    ex = result["gap_examples"][0]
    assert "series_id" in ex
    assert "n_missing_periods" in ex
    assert "first_missing" in ex
    assert "last_missing" in ex
    assert "sample_missing_dates" in ex
    assert "artifact_path" in result


def test_detect_temporal_gaps_raises_if_datetime_col_missing(tmp_path):
    df = pd.DataFrame({"sales": [1.0, 2.0], "store_id": ["S01", "S01"]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="not found"):
        detect_temporal_gaps.invoke({
            "dataset_path": str(path),
            "datetime_col": "date",
            "series_id_cols": ["store_id"],
            "frequency": "D",
            "target_column": "sales",
        })


def test_detect_temporal_gaps_raises_if_datetime_null(tmp_path):
    df = pd.DataFrame({
        "date": [None, "2024-01-02"],
        "sales": [1.0, 2.0],
        "store_id": ["S01", "S01"],
    })
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="null"):
        detect_temporal_gaps.invoke({
            "dataset_path": str(path),
            "datetime_col": "date",
            "series_id_cols": ["store_id"],
            "frequency": "D",
            "target_column": "sales",
        })


def test_detect_temporal_gaps_raises_if_series_id_null(tmp_path):
    df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "sales": [1.0, 2.0],
        "store_id": [None, "S01"],
    })
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="null"):
        detect_temporal_gaps.invoke({
            "dataset_path": str(path),
            "datetime_col": "date",
            "series_id_cols": ["store_id"],
            "frequency": "D",
            "target_column": "sales",
        })


def test_detect_temporal_gaps_raises_if_target_missing(tmp_path):
    df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02"],
        "store_id": ["S01", "S01"],
    })
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="target_column"):
        detect_temporal_gaps.invoke({
            "dataset_path": str(path),
            "datetime_col": "date",
            "series_id_cols": ["store_id"],
            "frequency": "D",
            "target_column": "sales",
        })


def test_detect_temporal_gaps_raises_if_invalid_frequency(tmp_path, daily_series_csv):
    with pytest.raises(ValueError, match="frequency"):
        detect_temporal_gaps.invoke({
            "dataset_path": str(daily_series_csv),
            "datetime_col": "date",
            "series_id_cols": ["store_id"],
            "frequency": "INVALID_FREQ",
            "target_column": "sales",
        })


def test_detect_temporal_gaps_raises_if_duplicates(tmp_path):
    df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-01"],
        "sales": [1.0, 2.0],
        "store_id": ["S01", "S01"],
    })
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        detect_temporal_gaps.invoke({
            "dataset_path": str(path),
            "datetime_col": "date",
            "series_id_cols": ["store_id"],
            "frequency": "D",
            "target_column": "sales",
        })


def test_detect_temporal_gaps_writes_artifact(tmp_path, daily_series_csv):
    artifact = tmp_path / "gaps.json"
    detect_temporal_gaps.invoke({
        "dataset_path": str(daily_series_csv),
        "datetime_col": "date",
        "series_id_cols": ["store_id"],
        "frequency": "D",
        "target_column": "sales",
        "output_path": str(artifact),
    })
    assert artifact.exists()
    import json as _json
    data = _json.loads(artifact.read_text())
    assert "gaps" in data


def test_detect_temporal_gaps_default_artifact_uses_stem(tmp_path, daily_series_csv):
    # Default artifact path includes dataset stem to avoid overwriting between runs
    result = json.loads(detect_temporal_gaps.invoke({
        "dataset_path": str(daily_series_csv),
        "datetime_col": "date",
        "series_id_cols": ["store_id"],
        "frequency": "D",
        "target_column": "sales",
    }))
    assert "series" in result["artifact_path"]  # stem of "series.csv"
```

- [ ] **Step 2: Run tests to verify they fail**

```
uv run pytest tests/test_tools/test_data_tools.py -k "detect_temporal" -v
```

Expected: FAIL with ImportError or NameError

- [ ] **Step 3: Implement `detect_temporal_gaps`**

Add to `src/mlops_agents/tools/data_tools.py` (before `impute_missing_values`). Add the import at the top of the file:

```python
from pandas.tseries.frequencies import to_offset
```

Then add the tool:

```python
@tool
def detect_temporal_gaps(
    dataset_path: str,
    datetime_col: str,
    series_id_cols: list[str],
    frequency: str,
    target_column: str,
    output_path: str = "",
) -> str:
    """Validate critical forecasting keys and detect missing time periods per series.

    Raises ValueError immediately if:
    - datetime_col does not exist or has nulls/unparseable values
    - any series_id_col has nulls
    - target_column is not present in the dataset
    - frequency is not a valid pandas offset alias
    - duplicate (series_id, datetime) pairs are found

    Writes a full gap report JSON artifact and returns a compact summary.

    Args:
        dataset_path: Path to the sorted CSV.
        datetime_col: Name of the datetime column.
        series_id_cols: Columns that identify each individual series.
        frequency: Pandas offset alias for expected cadence (e.g. "D", "W", "MS").
        target_column: Name of the target column (existence is validated here).
        output_path: Path to write the full gap report JSON artifact.

    Returns:
        JSON with {has_gaps, total_missing_periods, n_series_with_gaps,
        gap_examples (up to 5), artifact_path}.
    """
    path = Path(dataset_path)
    if not path.exists():
        return json.dumps({"error": f"File not found: {dataset_path}"})

    df = pd.read_csv(path)

    # Validate datetime_col exists before parsing
    if datetime_col not in df.columns:
        raise ValueError(f"datetime_col '{datetime_col}' not found in dataset")

    df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")

    # Critical key validation
    if df[datetime_col].isnull().any():
        raise ValueError(
            f"datetime_col '{datetime_col}' has {int(df[datetime_col].isnull().sum())} "
            "null or unparseable value(s)"
        )
    for col in series_id_cols:
        if col not in df.columns:
            raise ValueError(f"series_id_col '{col}' not found in dataset")
        if df[col].isnull().any():
            raise ValueError(
                f"series_id_col '{col}' has {int(df[col].isnull().sum())} null value(s)"
            )
    if target_column not in df.columns:
        raise ValueError(f"target_column '{target_column}' not found in dataset")

    # Validate frequency alias
    offset = to_offset(frequency)
    if offset is None:
        raise ValueError(f"Invalid pandas frequency alias: {frequency!r}")

    # Duplicate detection
    key_cols = [datetime_col] + list(series_id_cols)
    dup_count = int(df.duplicated(subset=key_cols).sum())
    if dup_count > 0:
        raise ValueError(
            f"Found {dup_count} duplicate (series_id, datetime) pair(s) — "
            "each (series, timestamp) must be unique"
        )

    # Gap detection
    def _gaps_for_group(grp: pd.DataFrame, series_id: dict) -> dict | None:
        actual = set(grp[datetime_col])
        expected = pd.date_range(
            grp[datetime_col].min(), grp[datetime_col].max(), freq=frequency
        )
        missing = sorted(set(expected) - actual)
        if not missing:
            return None
        return {
            "series_id": series_id,
            "n_missing_periods": len(missing),
            "first_missing": str(missing[0].date()),
            "last_missing": str(missing[-1].date()),
            "all_missing_dates": [str(d.date()) for d in missing],
            "sample_missing_dates": [str(d.date()) for d in missing[:3]],
        }

    all_gaps: list[dict] = []
    if series_id_cols:
        for keys, grp in df.groupby(series_id_cols):
            if not isinstance(keys, tuple):
                keys = (keys,)
            gap = _gaps_for_group(grp, dict(zip(series_id_cols, keys)))
            if gap:
                all_gaps.append(gap)
    else:
        gap = _gaps_for_group(df, {})
        if gap:
            all_gaps.append(gap)

    all_gaps.sort(key=lambda g: g["n_missing_periods"], reverse=True)
    total_missing = sum(g["n_missing_periods"] for g in all_gaps)

    gap_examples = [
        {k: v for k, v in g.items() if k != "all_missing_dates"}
        for g in all_gaps[:5]
    ]

    # Write full artifact (stem-based path to avoid overwriting between runs)
    artifact_path = output_path or f"artifacts/temporal_gaps_{path.stem}.json"
    Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
    Path(artifact_path).write_text(
        json.dumps(
            {"gaps": all_gaps, "total_missing_periods": total_missing}, default=str
        )
    )

    result = {
        "has_gaps": total_missing > 0,
        "total_missing_periods": total_missing,
        "n_series_with_gaps": len(all_gaps),
        "gap_examples": gap_examples,
        "artifact_path": artifact_path,
    }
    logger.info(
        f"Gap detection: {total_missing} missing periods across {len(all_gaps)} series"
    )
    return json.dumps(result, default=str)
```

- [ ] **Step 4: Run tests to verify they pass**

```
uv run pytest tests/test_tools/test_data_tools.py -k "detect_temporal" -v
```

Expected: all 11 tests PASS

- [ ] **Step 5: Run full test file to check for regressions**

```
uv run pytest tests/test_tools/test_data_tools.py -v
```

Expected: all tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/mlops_agents/tools/data_tools.py tests/test_tools/test_data_tools.py
git commit -m "feat: add detect_temporal_gaps tool — key validation, gap detection, artifact"
```

---

## Task 4: Register new tools in `data_agent.py`

**Files:**
- Modify: `src/mlops_agents/agents/data_agent.py`

- [ ] **Step 1: Update `data_agent.py`**

Replace the entire contents of `src/mlops_agents/agents/data_agent.py` with:

```python
"""Data Validation Agent — validates datasets before they enter the pipeline."""

from langchain.agents import create_agent

from mlops_agents.prompts import get_prompt
from mlops_agents.tools.data_tools import (
    apply_column_mapping,
    check_missing_values,
    detect_temporal_gaps,
    impute_missing_values,
    load_dataset,
    merge_datasets,
    parse_datetime_column,
    validate_against_schema,
)
from mlops_agents.tools.evidently_tools import check_data_drift, check_data_quality
from mlops_agents.utils.llm import get_llm


def build_data_agent():
    """Build and return the data validation react agent."""
    return create_agent(
        model=get_llm("data_validator"),
        tools=[
            load_dataset,
            merge_datasets,
            apply_column_mapping,
            validate_against_schema,
            check_missing_values,
            check_data_quality,
            check_data_drift,
            impute_missing_values,
            parse_datetime_column,
            detect_temporal_gaps,
        ],
        name="data_validator",
        system_prompt=get_prompt("data_agent").template,
    )
```

- [ ] **Step 2: Verify import succeeds**

```
uv run python -c "from mlops_agents.agents.data_agent import build_data_agent; print('OK')"
```

Expected: prints `OK`

- [ ] **Step 3: Run unit tests**

```
uv run pytest tests/ -m "not integration" -v
```

Expected: all unit tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/mlops_agents/agents/data_agent.py
git commit -m "feat: register parse_datetime_column and detect_temporal_gaps in data agent"
```

---

## Task 5: Update `data_agent.yaml` prompt with forecasting branch

**Files:**
- Modify: `src/mlops_agents/prompts/data_agent.yaml`

- [ ] **Step 1: Replace `data_agent.yaml` with the updated prompt**

Replace the entire contents of `src/mlops_agents/prompts/data_agent.yaml` with:

```yaml

_type: "prompt"
input_variables: []
template: |
  You are a Data Validation Specialist responsible for gating the MLOps pipeline.

  **AUTONOMY RULE: Never ask the user questions. Never ask for confirmation before
  calling a tool or modifying a file. Complete the full task and stop. The pipeline
  is fully automated — there is no one to answer your questions.**

  Your context message contains:
  - "Raw files": a JSON list of raw CSV file paths provided by the user
  - "Schema path": the full path to the target schema JSON file
  - "Target schema": the full schema JSON defining canonical columns, types,
    constraints, problem_type, target_column, datetime_column, series_id_columns,
    frequency, and mapping hints

  Your job is to merge the raw files into the canonical dataset, fix nullable
  violations automatically, validate all constraints, and report the result clearly.

  TOOLS:
  - load_dataset: Load and summarise a single CSV.
  - merge_datasets: Join multiple CSVs on a common key column.
  - apply_column_mapping: Rename/drop columns to match the canonical schema.
  - validate_against_schema: Check all schema constraints.
  - check_missing_values: Compute missing value statistics per column.
  - parse_datetime_column: Parse a string column as datetime and sort the dataset.
    Pass series_id_cols from the schema so multi-series data is sorted correctly.
    REQUIRED as step 1 for forecasting datasets.
  - detect_temporal_gaps: Validate critical forecasting keys (datetime column exists
    and has no nulls, series_id columns have no nulls, target column exists, frequency
    is valid, no duplicate keys), detect missing periods, and write a gap report
    artifact. REQUIRED as step 2 for forecasting datasets.
  - impute_missing_values: Fill missing values. Always pass problem_type and
    target_column. For classification/regression: rows with missing target are
    dropped; other columns get mean/mode imputation. For forecasting also pass
    datetime_column, series_id_columns, and max_interpolation_gap=3. The tool
    protects datetime and series_id columns (raises if null), applies short-gap
    linear interpolation to the target only (only gaps ≤ max_interpolation_gap that
    are bounded by known values on both sides), and applies dtype-aware
    forward-fill/interpolation to exogenous features.
  - check_data_quality: Run an Evidently AI quality report.
  - check_data_drift: Compare two CSVs for statistical drift (optional).

  PROCESS (classification or regression):
  1. Call load_dataset on EACH raw file to inspect its column names and types.
  2. Read the schema. Use "mapping_hint" and "is_key": true to decide which file
     contains which canonical columns and which column is the join key.
  3. Call merge_datasets with your join specification. Stop and report if any file
     lacks a matching key column.
  4. Call apply_column_mapping on the merged file. Use "data/processed/<schema_name>.csv"
     as output path.
  5. Call validate_against_schema on the canonical output.
  5b. If nullable violations are found, call impute_missing_values with
      problem_type and target_column, then call validate_against_schema once more.
      Do not repeat imputation more than once. Report remaining violations and stop
      if validation still fails.
  6. Call check_data_quality for an Evidently summary.

  PROCESS (forecasting — when schema.problem_type == "forecasting"):
  1. Call load_dataset on EACH raw file.
  2. Merge and map columns as above (steps 2–4 from classification process).
  3. Call parse_datetime_column with datetime_column and series_id_columns from the
     schema. Abort if it raises — null or unparseable timestamps are blocking errors.
  4. Call detect_temporal_gaps with datetime_column, series_id_columns, frequency,
     and target_column from the schema. Abort if it raises — broken keys or duplicates
     are blocking errors. Note the gap summary for reporting.
  5. Call impute_missing_values with problem_type="forecasting", target_column,
     datetime_column, series_id_columns, and max_interpolation_gap=3. Check
     target_large_gaps in the result — report these as warnings but do not treat them
     as blocking failures unless the target column is entirely missing.
  6. Call validate_against_schema.
  7. Call check_data_quality.

  REPORTING:
  Report clearly at the end:
  - PASSED or FAILED
  - Which files were merged on which key columns
  - Which raw columns were mapped to which canonical names
  - Any constraint violations with detail
  - If imputation was applied: which columns, what strategy, any warnings (e.g. rows
    dropped due to missing target)
  - For forecasting: gap summary (has_gaps, total_missing_periods, series affected)
    and any large target gaps flagged as warnings

  Be specific. The supervisor uses your output to decide whether to proceed to
  training. If validation fails after auto-fix, explain what the data engineer
  needs to fix in the source files — do not suggest they reply to you.
```

- [ ] **Step 2: Verify the prompt loads without errors**

```
uv run python -c "
from mlops_agents.prompts import get_prompt
p = get_prompt('data_agent')
print('template length:', len(p.template))
print('OK')
"
```

Expected: prints template length and `OK`

- [ ] **Step 3: Run full unit test suite**

```
uv run pytest tests/ -m "not integration" -v
```

Expected: all unit tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/mlops_agents/prompts/data_agent.yaml
git commit -m "feat: add forecasting branch to data_agent prompt with 7-step temporal validation flow"
```
