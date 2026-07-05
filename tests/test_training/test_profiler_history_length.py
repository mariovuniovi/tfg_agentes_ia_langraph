"""Tests for the new history_length field and DatasetProfile typing."""
from pathlib import Path

import pandas as pd

from mlops_agents.training.profiler import DatasetProfile, build_dataset_profile


def _write_csv(rows: int, tmp_path: Path) -> Path:
    dates = pd.date_range("2020-01-01", periods=rows, freq="W")
    df = pd.DataFrame({"date": dates, "target": range(rows)})
    p = tmp_path / "ts.csv"
    df.to_csv(p, index=False)
    return p


def test_profile_is_pydantic_model_with_attribute_access(tmp_path):
    csv = _write_csv(60, tmp_path)
    profile = build_dataset_profile(
        csv, {"problem_type": "forecasting", "target_column": "target",
              "datetime_column": "date", "frequency": "W", "forecast_horizon": 4,
              "series_id_columns": []},
    )
    assert isinstance(profile, DatasetProfile)
    assert profile.history_length is not None  # set for forecasting


def test_history_length_short_for_50_rows(tmp_path):
    csv = _write_csv(50, tmp_path)
    profile = build_dataset_profile(
        csv, {"problem_type": "forecasting", "target_column": "target",
              "datetime_column": "date", "frequency": "W", "forecast_horizon": 4,
              "series_id_columns": []},
    )
    assert profile.history_length == "very_short"


def test_history_length_medium_for_500_rows(tmp_path):
    csv = _write_csv(500, tmp_path)
    profile = build_dataset_profile(
        csv, {"problem_type": "forecasting", "target_column": "target",
              "datetime_column": "date", "frequency": "W", "forecast_horizon": 4,
              "series_id_columns": []},
    )
    assert profile.history_length == "medium"


def test_history_length_none_for_tabular(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "target": [0, 1, 0]})
    csv = tmp_path / "tab.csv"
    df.to_csv(csv, index=False)
    profile = build_dataset_profile(
        csv, {"problem_type": "classification", "target_column": "target"},
    )
    assert profile.history_length is None


def test_profile_can_serialize_to_json(tmp_path):
    csv = _write_csv(60, tmp_path)
    profile = build_dataset_profile(
        csv, {"problem_type": "forecasting", "target_column": "target",
              "datetime_column": "date", "frequency": "W", "forecast_horizon": 4,
              "series_id_columns": []},
    )
    s = profile.model_dump_json()
    assert "history_length" in s
