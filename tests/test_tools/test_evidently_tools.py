"""Unit tests for Evidently AI tools — Evidently is mocked for speed and stability.

Mocking strategy: patch 'Report', 'DataSummaryPreset', and 'DataDriftPreset' at the
module level in evidently_tools (they are imported at module top level, so the
standard @patch approach works). The real smoke tests exercise the actual Evidently
API with a small CSV.
"""

import json
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# check_data_quality
# ---------------------------------------------------------------------------

@patch("mlops_agents.tools.evidently_tools.DataSummaryPreset")
@patch("mlops_agents.tools.evidently_tools.Report")
@patch("mlops_agents.tools.evidently_tools.pd")
def test_check_data_quality_returns_passed_key(mock_pd, mock_report_cls, mock_preset, sample_csv):
    mock_df = MagicMock()
    mock_df.__len__ = lambda _: 5
    mock_df.columns.__len__ = lambda _: 3
    mock_pd.read_csv.return_value = mock_df

    mock_report_instance = MagicMock()
    mock_result = MagicMock()
    mock_result.dump_dict.return_value = {"summary": "ok"}
    mock_report_instance.run.return_value = mock_result
    mock_report_cls.return_value = mock_report_instance

    from mlops_agents.tools.evidently_tools import check_data_quality

    result = json.loads(check_data_quality.invoke({"dataset_path": str(sample_csv)}))

    assert "passed" in result
    assert "report" in result


@patch("mlops_agents.tools.evidently_tools.DataSummaryPreset")
@patch("mlops_agents.tools.evidently_tools.Report")
@patch("mlops_agents.tools.evidently_tools.pd")
def test_check_data_quality_includes_row_and_column_count(mock_pd, mock_report_cls, mock_preset, sample_csv):
    mock_df = MagicMock()
    mock_df.__len__ = lambda _: 5

    mock_columns = MagicMock()
    mock_columns.__len__ = lambda _: 3
    mock_df.columns = mock_columns

    mock_pd.read_csv.return_value = mock_df

    mock_report_instance = MagicMock()
    mock_result = MagicMock()
    mock_result.dump_dict.return_value = {}
    mock_report_instance.run.return_value = mock_result
    mock_report_cls.return_value = mock_report_instance

    from mlops_agents.tools.evidently_tools import check_data_quality

    result = json.loads(check_data_quality.invoke({"dataset_path": str(sample_csv)}))

    assert "row_count" in result
    assert "column_count" in result


def test_check_data_quality_with_real_evidently(sample_csv):
    """Smoke test: run against real Evidently with a real CSV."""
    from mlops_agents.tools.evidently_tools import check_data_quality

    result = json.loads(check_data_quality.invoke({"dataset_path": str(sample_csv)}))

    assert "passed" in result
    assert "report" in result
    assert "row_count" in result
    assert result["row_count"] == 5


# ---------------------------------------------------------------------------
# check_data_drift
# ---------------------------------------------------------------------------

@patch("mlops_agents.tools.evidently_tools.DataDriftPreset")
@patch("mlops_agents.tools.evidently_tools.Report")
@patch("mlops_agents.tools.evidently_tools.pd")
def test_check_data_drift_returns_drift_detected_key(mock_pd, mock_report_cls, mock_preset, sample_csv):
    mock_df = MagicMock()
    mock_pd.read_csv.return_value = mock_df

    mock_report_instance = MagicMock()
    mock_result = MagicMock()
    mock_result.dump_dict.return_value = {"drift": "none"}
    mock_report_instance.run.return_value = mock_result
    mock_report_cls.return_value = mock_report_instance

    from mlops_agents.tools.evidently_tools import check_data_drift

    result = json.loads(check_data_drift.invoke({
        "current_path": str(sample_csv),
        "reference_path": str(sample_csv),
    }))

    assert "drift_detected" in result
    assert "report" in result


@patch("mlops_agents.tools.evidently_tools.DataDriftPreset")
@patch("mlops_agents.tools.evidently_tools.Report")
@patch("mlops_agents.tools.evidently_tools.pd")
def test_check_data_drift_default_is_no_drift(mock_pd, mock_report_cls, mock_preset, sample_csv):
    """drift_detected should default to False — the LLM agent interprets the report."""
    mock_df = MagicMock()
    mock_pd.read_csv.return_value = mock_df

    mock_report_instance = MagicMock()
    mock_result = MagicMock()
    mock_result.dump_dict.return_value = {}
    mock_report_instance.run.return_value = mock_result
    mock_report_cls.return_value = mock_report_instance

    from mlops_agents.tools.evidently_tools import check_data_drift

    result = json.loads(check_data_drift.invoke({
        "current_path": str(sample_csv),
        "reference_path": str(sample_csv),
    }))

    assert result["drift_detected"] is False


def test_check_data_drift_with_real_evidently(sample_csv):
    """Smoke test: identical reference and current data should produce a valid report."""
    from mlops_agents.tools.evidently_tools import check_data_drift

    result = json.loads(check_data_drift.invoke({
        "current_path": str(sample_csv),
        "reference_path": str(sample_csv),
    }))

    assert "drift_detected" in result
    assert "report" in result
