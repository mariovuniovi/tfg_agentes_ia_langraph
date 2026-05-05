"""Deterministic data loading and schema validation tools.

These tools are pure Python — no LLM involved. They run inside the
data validation agent's tool-calling loop to gather facts; the agent
then interprets the results.
"""

import json
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

from mlops_agents.config.constants import MAX_DRIFT_SCORE
from mlops_agents.utils.logging import get_logger

logger = get_logger(__name__)


@tool
def load_dataset(dataset_path: str) -> str:
    """Load a CSV dataset and return a JSON summary of its shape and columns.

    Args:
        dataset_path: Absolute or relative path to the CSV file.

    Returns:
        JSON string with row_count, column_names, dtypes, and sample head.
    """
    path = Path(dataset_path)
    if not path.exists():
        return json.dumps({"error": f"File not found: {dataset_path}"})

    df = pd.read_csv(path)
    summary = {
        "row_count": len(df),
        "column_count": len(df.columns),
        "column_names": df.columns.tolist(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "head": df.head(3).to_dict(orient="records"),
    }
    logger.info(f"Loaded dataset: {path.name} ({len(df)} rows, {len(df.columns)} columns)")
    return json.dumps(summary, default=str)


@tool
def check_missing_values(dataset_path: str) -> str:
    """Compute missing value statistics for every column.

    Args:
        dataset_path: Path to the CSV file.

    Returns:
        JSON with per-column missing counts and percentages, plus overall summary.
    """
    df = pd.read_csv(dataset_path)
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)

    per_column = {
        col: {"count": int(missing[col]), "pct": float(pct[col])}
        for col in df.columns
        if missing[col] > 0
    }

    result = {
        "total_rows": len(df),
        "columns_with_missing": len(per_column),
        "max_missing_pct": float(pct.max()),
        "per_column": per_column,
        "passed_threshold": float(pct.max()) < (MAX_DRIFT_SCORE * 100),
    }
    return json.dumps(result)


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
    violations: list[dict[str, str]] = []

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

    mapping: dict[str, str] = json.loads(mapping_json)
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
    file_specs: list[dict[str, str]] = spec["files"]

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
    internal gaps in the target are interpolated (<= max_interpolation_gap, bounded on
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
