"""Tests for ValidationStrategy, ExogStrategySettings, ForecastingSettings."""
import pytest
from pydantic import ValidationError

from mlops_agents.contracts.training import (
    ExogStrategySettings,
    ForecastingSettings,
    ValidationStrategy,
)


def test_validation_strategy_defaults_single_split():
    s = ValidationStrategy(horizon=12)
    assert s.type == "single_split"
    assert s.n_folds == 1
    assert s.step_size is None
    assert s.window_size is None


def test_validation_strategy_rejects_unknown_type():
    with pytest.raises(ValidationError):
        ValidationStrategy(type="nonsense", horizon=12)


def test_validation_strategy_rolling_with_window():
    s = ValidationStrategy(
        type="rolling_window", n_folds=3, horizon=12, step_size=12, window_size=60
    )
    assert s.window_size == 60


def test_exog_strategy_settings_empty_defaults():
    e = ExogStrategySettings()
    assert e.per_column == {}
    assert e.default_unknown_future == "naive_carry"


def test_exog_strategy_settings_per_column_rejects_unknown_value():
    with pytest.raises(ValidationError):
        ExogStrategySettings(per_column={"oil": "magic"})


def test_forecasting_settings_compose():
    fs = ForecastingSettings(
        validation_strategy=ValidationStrategy(horizon=6),
        exog_strategies=ExogStrategySettings(per_column={"oil": "auto_arima"}),
    )
    assert fs.validation_strategy.n_folds == 1
    assert fs.exog_strategies.per_column["oil"] == "auto_arima"


def test_validation_strategy_rejects_non_positive_horizon_and_folds():
    with pytest.raises(ValidationError):
        ValidationStrategy(horizon=0)
    with pytest.raises(ValidationError):
        ValidationStrategy(horizon=-1)
    with pytest.raises(ValidationError):
        ValidationStrategy(n_folds=0, horizon=10)
