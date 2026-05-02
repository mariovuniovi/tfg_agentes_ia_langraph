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
from mlops_agents.config.settings import settings
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


@tool
def impute_missing_values(path: str) -> str:
    """Impute missing values in a canonical CSV using strategies from settings.

    Numeric columns (float64, int64): uses settings.imputation_strategy_numeric
    Categorical columns (object): uses settings.imputation_strategy_categorical

    Writes the result back to the same path (in-place).

    Args:
        path: Path to the canonical CSV file to impute.

    Returns:
        JSON with {output_path, imputed_columns} where each imputed column
        maps to {strategy, fill_value, rows_affected}.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        return json.dumps({"error": f"File not found: {path}"})

    df = pd.read_csv(csv_path)
    imputed: dict[str, dict] = {}

    numeric_strategy = settings.imputation_strategy_numeric
    categorical_strategy = settings.imputation_strategy_categorical

    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        if null_count == 0:
            continue

        if df[col].dtype in ("float64", "int64"):
            if numeric_strategy == "mean":
                fill_value = float(df[col].mean())
            elif numeric_strategy == "median":
                fill_value = float(df[col].median())
            else:  # "zero"
                fill_value = 0.0
            df[col] = df[col].fillna(fill_value)
            imputed[col] = {"strategy": numeric_strategy, "fill_value": fill_value, "rows_affected": null_count}

        elif df[col].dtype == object:
            if categorical_strategy == "mode":
                fill_value = str(df[col].mode().iloc[0]) if not df[col].mode().empty else "unknown"
                df[col] = df[col].fillna(fill_value)
                imputed[col] = {"strategy": "mode", "fill_value": fill_value, "rows_affected": null_count}
            elif categorical_strategy == "unknown":
                df[col] = df[col].fillna("unknown")
                imputed[col] = {"strategy": "unknown", "fill_value": "unknown", "rows_affected": null_count}
            else:  # "drop_row"
                df = df.dropna(subset=[col])
                imputed[col] = {"strategy": "drop_row", "fill_value": None, "rows_affected": null_count}

    df.to_csv(csv_path, index=False)
    logger.info(f"Imputed {len(imputed)} column(s) in {csv_path.name}")
    return json.dumps({"output_path": str(csv_path), "imputed_columns": imputed}, default=str)
