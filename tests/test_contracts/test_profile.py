"""Tests for DatasetProfile Pydantic schema."""
import pytest
from pydantic import ValidationError

from mlops_agents.contracts.profile import DatasetProfile


def test_classification_profile_valid():
    p = DatasetProfile(
        problem_type="classification",
        n_rows="small", n_features="small", missing_rate="none",
        n_categorical_features="none", n_numerical_features="few",
        n_classes="binary", class_balance="balanced",
    )
    assert p.problem_type == "classification"
    assert p.n_classes == "binary"


def test_regression_profile_valid():
    p = DatasetProfile(
        problem_type="regression",
        n_rows="medium", n_features="small", missing_rate="low",
        n_categorical_features="none", n_numerical_features="few",
        target_distribution="skewed",
    )
    assert p.target_distribution == "skewed"


def test_forecasting_profile_valid():
    p = DatasetProfile(
        problem_type="forecasting",
        n_rows="medium", n_features="small", missing_rate="none",
        n_categorical_features="none", n_numerical_features="few",
        n_series="single", history_length="medium", frequency="MS",
        horizon_difficulty="short", forecast_horizon_raw=12,
        exogenous_features_available=False,
        seasonality_detected=True, trend_detected=False, stationarity=False,
    )
    assert p.n_series == "single"
    assert p.seasonality_detected is True


def test_invalid_n_rows_bucket_rejected():
    with pytest.raises(ValidationError):
        DatasetProfile(
            problem_type="classification",
            n_rows="gigantic",
            n_features="small", missing_rate="none",
            n_categorical_features="none", n_numerical_features="few",
        )


def test_schema_version_default():
    p = DatasetProfile(
        problem_type="regression",
        n_rows="small", n_features="small", missing_rate="none",
        n_categorical_features="none", n_numerical_features="few",
    )
    assert p.schema_version == 1
