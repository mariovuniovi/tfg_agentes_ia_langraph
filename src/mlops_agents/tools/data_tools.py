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
def validate_schema(dataset_path: str, expected_columns: str) -> str:
    """Check that the dataset has all expected columns with correct types.

    Args:
        dataset_path: Path to the CSV file.
        expected_columns: JSON array of column name strings that must be present.

    Returns:
        JSON with 'valid' bool, 'missing_columns', and 'extra_columns' lists.
    """
    df = pd.read_csv(dataset_path)
    expected = json.loads(expected_columns)
    actual = set(df.columns.tolist())
    expected_set = set(expected)

    result = {
        "valid": expected_set.issubset(actual),
        "missing_columns": list(expected_set - actual),
        "extra_columns": list(actual - expected_set),
        "total_columns": len(actual),
    }
    return json.dumps(result)


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
