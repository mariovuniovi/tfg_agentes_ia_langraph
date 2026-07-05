"""Tests for check_data_quality (now in data_tools)."""
import json


def test_check_data_quality_returns_passed_key(sample_csv):
    from mlops_agents.tools.data_tools import check_data_quality
    result = json.loads(check_data_quality.invoke({"dataset_path": str(sample_csv)}))
    assert "passed" in result


def test_check_data_quality_includes_row_and_column_count(sample_csv):
    from mlops_agents.tools.data_tools import check_data_quality
    result = json.loads(check_data_quality.invoke({"dataset_path": str(sample_csv)}))
    assert "row_count" in result
    assert "column_count" in result
    assert result["row_count"] == 5


def test_check_data_quality_no_missing_clean_csv(sample_csv):
    from mlops_agents.tools.data_tools import check_data_quality
    result = json.loads(check_data_quality.invoke({"dataset_path": str(sample_csv)}))
    assert result["missing_values_total"] == 0
    assert result["duplicate_rows"] == 0
    assert result["passed"] is True
