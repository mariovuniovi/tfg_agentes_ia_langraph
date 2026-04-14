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
