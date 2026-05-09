"""Tests for the train/pool/test splitter."""

import json

import pandas as pd
import pytest

from mlops_agents.training.splitter import split_dataset


def _write_csv(tmp_path, df, name="data.csv"):
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


def test_classification_stratified_split(tmp_path):
    df = pd.DataFrame({"x": range(100), "target": [0] * 70 + [1] * 30})
    csv = _write_csv(tmp_path, df)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    train_pool, test, meta = split_dataset(
        csv, {"problem_type": "classification", "target_column": "target"},
        out_dir, test_size=0.2, random_state=42,
    )
    assert train_pool.exists()
    assert test.exists()
    meta_data = json.loads(meta.read_text())
    assert meta_data["split_kind"] == "stratified"
    train_df = pd.read_csv(train_pool)
    test_df = pd.read_csv(test)
    assert len(train_df) == 80
    assert len(test_df) == 20


def test_regression_random_shuffle_split(tmp_path):
    df = pd.DataFrame({"x": range(50), "y": [float(i) for i in range(50)]})
    csv = _write_csv(tmp_path, df)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    train_pool, test, meta = split_dataset(
        csv, {"problem_type": "regression", "target_column": "y"},
        out_dir, test_size=0.2, random_state=42,
    )
    meta_data = json.loads(meta.read_text())
    assert meta_data["split_kind"] == "random_shuffle"


def test_forecasting_temporal_split_single_series(tmp_path):
    dates = pd.date_range("2020-01-01", periods=60, freq="MS")
    df = pd.DataFrame({"ds": dates, "y": range(60)})
    csv = _write_csv(tmp_path, df)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    train_pool, test, meta = split_dataset(
        csv,
        {
            "problem_type": "forecasting", "target_column": "y",
            "datetime_column": "ds", "series_id_columns": [],
            "frequency": "MS", "forecast_horizon": 12,
        },
        out_dir, test_size=0.2, random_state=42,
    )
    train_df = pd.read_csv(train_pool, parse_dates=["ds"])
    test_df = pd.read_csv(test, parse_dates=["ds"])
    assert len(test_df) == 12
    assert len(train_df) == 48
    assert train_df["ds"].max() < test_df["ds"].min()
    meta_data = json.loads(meta.read_text())
    assert meta_data["split_kind"] == "temporal_per_series"


def test_forecasting_drops_short_series_majority_minority(tmp_path):
    """If < 50% of series are too short, drop them and continue."""
    rows = []
    for sid in ["a", "b", "c"]:           # a, b, c all OK (60 each)
        rows += [{"sid": sid, "ds": d, "y": float(i)}
                 for i, d in enumerate(pd.date_range("2020-01-01", periods=60, freq="MS"))]
    rows += [{"sid": "tiny", "ds": d, "y": float(i)}
             for i, d in enumerate(pd.date_range("2020-01-01", periods=5, freq="MS"))]
    df = pd.DataFrame(rows)
    csv = _write_csv(tmp_path, df)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    _, _, meta = split_dataset(
        csv,
        {
            "problem_type": "forecasting", "target_column": "y",
            "datetime_column": "ds", "series_id_columns": ["sid"],
            "frequency": "MS", "forecast_horizon": 12,
        },
        out_dir, test_size=0.2, random_state=42,
    )
    meta_data = json.loads(meta.read_text())
    assert meta_data["n_series_dropped"] == 1


def test_forecasting_majority_too_short_raises(tmp_path):
    """If > 50% of series are too short, raise ValueError."""
    rows = []
    for sid in ["a", "b", "c"]:           # all too short (10 each)
        rows += [{"sid": sid, "ds": d, "y": float(i)}
                 for i, d in enumerate(pd.date_range("2020-01-01", periods=10, freq="MS"))]
    df = pd.DataFrame(rows)
    csv = _write_csv(tmp_path, df)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with pytest.raises(ValueError, match="too short"):
        split_dataset(
            csv,
            {
                "problem_type": "forecasting", "target_column": "y",
                "datetime_column": "ds", "series_id_columns": ["sid"],
                "frequency": "MS", "forecast_horizon": 12,
            },
            out_dir, test_size=0.2, random_state=42,
        )
