import pandas as pd

from mlops_agents.training.profiler import build_dataset_profile


def test_regression_profile_has_numeric_target_stats(tmp_path):
    csv = tmp_path / "r.csv"
    pd.DataFrame({"x": range(10), "target": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]}).to_csv(csv, index=False)
    profile = build_dataset_profile(csv, {"problem_type": "regression", "target_column": "target"})
    assert profile.target_mean == 5.5
    assert profile.target_min == 1.0
    assert profile.target_max == 10.0
    assert profile.target_std is not None and profile.target_std > 0


def test_classification_profile_has_none_target_stats(tmp_path):
    csv = tmp_path / "c.csv"
    pd.DataFrame({"x": range(6), "target": ["a", "b", "a", "b", "a", "b"]}).to_csv(csv, index=False)
    profile = build_dataset_profile(csv, {"problem_type": "classification", "target_column": "target"})
    assert profile.target_mean is None
    assert profile.target_std is None
    assert profile.target_min is None
    assert profile.target_max is None


def test_forecasting_profile_has_numeric_target_stats(tmp_path):
    csv = tmp_path / "f.csv"
    pd.DataFrame({"ds": pd.date_range("2024-01-01", periods=10, freq="D"),
                  "y": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]}).to_csv(csv, index=False)
    profile = build_dataset_profile(csv, {
        "problem_type": "forecasting",
        "target_column": "y",
        "datetime_column": "ds",
        "frequency": "D",
        "forecast_horizon": 7,
    })
    assert profile.target_mean == 55.0
    assert profile.target_min == 10.0
    assert profile.target_max == 100.0
