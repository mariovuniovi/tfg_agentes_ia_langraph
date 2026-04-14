"""Unit tests for data tools (deterministic — no LLM calls)."""

import json
from pathlib import Path

import pandas as pd
import pytest

from mlops_agents.tools.data_tools import check_missing_values, load_dataset, validate_schema


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------

def test_load_dataset_returns_row_count(sample_csv):
    result = json.loads(load_dataset.invoke({"dataset_path": str(sample_csv)}))
    assert result["row_count"] == 5


def test_load_dataset_returns_column_names(sample_csv):
    result = json.loads(load_dataset.invoke({"dataset_path": str(sample_csv)}))
    assert "feature_1" in result["column_names"]
    assert "feature_2" in result["column_names"]
    assert "target" in result["column_names"]


def test_load_dataset_returns_column_count(sample_csv):
    result = json.loads(load_dataset.invoke({"dataset_path": str(sample_csv)}))
    assert result["column_count"] == 3


def test_load_dataset_includes_head_rows(sample_csv):
    result = json.loads(load_dataset.invoke({"dataset_path": str(sample_csv)}))
    assert "head" in result
    assert isinstance(result["head"], list)
    assert len(result["head"]) == 3  # head(3) default


def test_load_dataset_returns_dtypes(sample_csv):
    result = json.loads(load_dataset.invoke({"dataset_path": str(sample_csv)}))
    assert "dtypes" in result
    assert "target" in result["dtypes"]


def test_load_dataset_returns_error_for_missing_file():
    result = json.loads(load_dataset.invoke({"dataset_path": "/nonexistent/path.csv"}))
    assert "error" in result


# ---------------------------------------------------------------------------
# validate_schema
# ---------------------------------------------------------------------------

def test_validate_schema_passes_when_all_columns_present(sample_csv):
    expected = json.dumps(["feature_1", "feature_2", "target"])
    result = json.loads(validate_schema.invoke({
        "dataset_path": str(sample_csv),
        "expected_columns": expected,
    }))
    assert result["valid"] is True
    assert result["missing_columns"] == []


def test_validate_schema_passes_with_subset_of_columns(sample_csv):
    """Subset is valid — all expected columns exist even if dataset has extras."""
    expected = json.dumps(["feature_1", "target"])
    result = json.loads(validate_schema.invoke({
        "dataset_path": str(sample_csv),
        "expected_columns": expected,
    }))
    assert result["valid"] is True


def test_validate_schema_fails_when_column_missing(sample_csv):
    expected = json.dumps(["feature_1", "nonexistent_col"])
    result = json.loads(validate_schema.invoke({
        "dataset_path": str(sample_csv),
        "expected_columns": expected,
    }))
    assert result["valid"] is False
    assert "nonexistent_col" in result["missing_columns"]


def test_validate_schema_reports_extra_columns(sample_csv):
    """Extra columns in dataset (not in expected) are reported but don't fail."""
    expected = json.dumps(["target"])
    result = json.loads(validate_schema.invoke({
        "dataset_path": str(sample_csv),
        "expected_columns": expected,
    }))
    assert result["valid"] is True
    assert "feature_1" in result["extra_columns"]
    assert "feature_2" in result["extra_columns"]


def test_validate_schema_returns_total_column_count(sample_csv):
    expected = json.dumps(["target"])
    result = json.loads(validate_schema.invoke({
        "dataset_path": str(sample_csv),
        "expected_columns": expected,
    }))
    assert result["total_columns"] == 3


# ---------------------------------------------------------------------------
# check_missing_values
# ---------------------------------------------------------------------------

def test_check_missing_values_no_missing(sample_csv):
    result = json.loads(check_missing_values.invoke({"dataset_path": str(sample_csv)}))
    assert result["columns_with_missing"] == 0
    assert result["max_missing_pct"] == 0.0


def test_check_missing_values_detects_missing(sample_csv_with_missing):
    result = json.loads(check_missing_values.invoke({"dataset_path": str(sample_csv_with_missing)}))
    assert result["columns_with_missing"] == 1
    assert "feature_2" in result["per_column"]
    assert result["per_column"]["feature_2"]["count"] == 2
    assert result["per_column"]["feature_2"]["pct"] == 40.0


def test_check_missing_values_returns_total_rows(sample_csv):
    result = json.loads(check_missing_values.invoke({"dataset_path": str(sample_csv)}))
    assert result["total_rows"] == 5


def test_check_missing_values_passed_threshold_false_when_high_missing(sample_csv_with_missing):
    """40% missing in feature_2 should fail the threshold (MAX_DRIFT_SCORE * 100 = 10%)."""
    result = json.loads(check_missing_values.invoke({"dataset_path": str(sample_csv_with_missing)}))
    assert result["passed_threshold"] is False


def test_check_missing_values_passed_threshold_true_when_clean(sample_csv):
    result = json.loads(check_missing_values.invoke({"dataset_path": str(sample_csv)}))
    assert result["passed_threshold"] is True
