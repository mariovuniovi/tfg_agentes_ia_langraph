"""Pydantic schema for the dataset profile — the retrieval join key for SP4/SP5."""
from __future__ import annotations
from typing import Literal
from pydantic import BaseModel


class DatasetProfile(BaseModel):
    model_config = {"extra": "allow"}

    schema_version: int = 1
    problem_type: Literal["classification", "regression", "forecasting"]
    # Universal
    n_rows: Literal["very_small", "small", "medium", "large"]
    n_features: Literal["small", "medium", "large"]
    missing_rate: Literal["none", "low", "medium", "high"]
    n_categorical_features: Literal["none", "few", "some", "many"]
    n_numerical_features: Literal["none", "few", "some", "many"]
    # Classification-only
    n_classes: Literal["binary", "small_multiclass", "many_classes"] | None = None
    class_balance: Literal["balanced", "moderately_imbalanced", "severely_imbalanced"] | None = None
    # Regression-only
    target_distribution: Literal["near_normal", "skewed", "heavy_tailed", "discrete_like"] | None = None
    # Forecasting-only
    n_series: Literal["single", "few", "moderate", "many"] | None = None
    history_length: Literal["very_short", "short", "medium", "long"] | None = None
    frequency: str | None = None
    horizon_difficulty: Literal["very_short", "short", "medium", "long"] | None = None
    forecast_horizon_raw: int | None = None
    exogenous_features_available: bool | None = None
    seasonality_detected: bool | None = None
    trend_detected: bool | None = None
    stationarity: bool | None = None
    expected_drift: Literal["low", "medium", "high"] | None = None
    exog_column_kind: str | None = None
    exog_future_availability: Literal["known_future", "unknown_future"] | None = None
    exog_kind: str | None = None
    # Numeric target stats (regression + forecasting only)
    target_mean: float | None = None
    target_std: float | None = None
    target_min: float | None = None
    target_max: float | None = None
