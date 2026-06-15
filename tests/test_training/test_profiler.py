"""Tests for build_dataset_profile."""

import pandas as pd

from mlops_agents.training.profiler import build_dataset_profile


def _write_csv(tmp_path, df, name="data.csv"):
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


def test_classification_profile_basic(tmp_path):
    df = pd.DataFrame({
        "f1": range(60),
        "f2": [0.1] * 60,
        "cat": (["a", "b"] * 30),
        "target": [0, 1] * 30,
    })
    csv = _write_csv(tmp_path, df)
    p = build_dataset_profile(csv, {"problem_type": "classification", "target_column": "target"})
    assert p.problem_type == "classification"
    assert p.n_rows == "very_small"
    assert p.n_classes == "binary"
    assert p.class_balance == "balanced"


def test_regression_profile_basic(tmp_path):
    df = pd.DataFrame({"x1": range(2000), "x2": [0.5] * 2000, "y": list(range(2000))})
    csv = _write_csv(tmp_path, df)
    p = build_dataset_profile(csv, {"problem_type": "regression", "target_column": "y"})
    assert p.problem_type == "regression"
    assert p.n_rows == "medium"


def test_classification_imbalance_severely(tmp_path):
    df = pd.DataFrame({"f1": range(60), "target": [0] * 50 + [1] * 10})
    csv = _write_csv(tmp_path, df)
    p = build_dataset_profile(csv, {"problem_type": "classification", "target_column": "target"})
    assert p.class_balance == "severely_imbalanced"


def test_missing_rate_low(tmp_path):
    df = pd.DataFrame({"f1": [1.0, 2.0, None, 4.0, 5.0] * 12, "target": [0, 1] * 30})
    csv = _write_csv(tmp_path, df)
    p = build_dataset_profile(csv, {"problem_type": "classification", "target_column": "target"})
    assert p.missing_rate in ("low", "medium")


def test_forecasting_profile_single_series(tmp_path):
    dates = pd.date_range("2020-01-01", periods=120, freq="MS")
    df = pd.DataFrame({"ds": dates, "y": range(120)})
    csv = _write_csv(tmp_path, df)
    p = build_dataset_profile(
        csv,
        {
            "problem_type": "forecasting",
            "target_column": "y",
            "datetime_column": "ds",
            "series_id_columns": [],
            "frequency": "MS",
            "forecast_horizon": 12,
        },
    )
    assert p.problem_type == "forecasting"
    assert p.n_series == "single"
    assert p.history_length in ("short", "medium")
    assert p.frequency == "MS"
    assert p.horizon_difficulty in ("very_short", "short", "medium", "long")
    assert isinstance(p.seasonality_detected, bool)
    assert isinstance(p.trend_detected, bool)
    assert isinstance(p.stationarity, bool)
