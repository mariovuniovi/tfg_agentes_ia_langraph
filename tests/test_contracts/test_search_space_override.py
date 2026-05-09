"""Tests for search_space_override validation against the model registry."""

import pytest

from mlops_agents.contracts.training import SearchParamOverride
from mlops_agents.training.override_validation import (
    narrow_search_space,
    validate_override,
)


def test_override_within_range_accepted():
    overrides = {
        "n_estimators": SearchParamOverride(low=200, high=500),
        "learning_rate": SearchParamOverride(low=0.01, high=0.1),
    }
    # No error raised
    validate_override("lightgbm_regressor", overrides)


def test_override_choices_subset_of_categorical_accepted():
    overrides = {"penalty": SearchParamOverride(choices=["l2"])}
    validate_override("logistic_regression", overrides)


def test_override_unknown_param_rejected():
    overrides = {"nonsense": SearchParamOverride(low=1, high=2)}
    with pytest.raises(ValueError, match="unknown.*parameter|not in registry"):
        validate_override("lightgbm_regressor", overrides)


def test_override_out_of_range_rejected():
    overrides = {"learning_rate": SearchParamOverride(low=0.001, high=10.0)}  # high > registry.high (0.2)
    with pytest.raises(ValueError, match="out of registry|disjoint|wider"):
        validate_override("lightgbm_regressor", overrides)


def test_override_choices_outside_categorical_rejected():
    overrides = {"penalty": SearchParamOverride(choices=["elasticnet"])}
    with pytest.raises(ValueError, match="not in registry"):
        validate_override("logistic_regression", overrides)


def test_override_categorical_with_low_high_rejected():
    overrides = {"penalty": SearchParamOverride(low=0, high=1)}
    with pytest.raises(ValueError, match="categorical"):
        validate_override("logistic_regression", overrides)


def test_narrow_search_space_collapses_int_to_categorical_via_choices():
    overrides = {"n_estimators": SearchParamOverride(choices=[300, 500, 800])}
    narrowed = narrow_search_space("lightgbm_regressor", overrides)
    n_param = narrowed.params["n_estimators"]
    assert n_param.type == "categorical"
    assert n_param.choices == [300, 500, 800]


def test_narrow_search_space_keeps_unmodified_params():
    """Params not in override keep registry defaults."""
    overrides = {"learning_rate": SearchParamOverride(low=0.01, high=0.05)}
    narrowed = narrow_search_space("lightgbm_regressor", overrides)
    # learning_rate narrowed
    assert narrowed.params["learning_rate"].low == 0.01
    assert narrowed.params["learning_rate"].high == 0.05
    # n_estimators unchanged from registry
    assert narrowed.params["n_estimators"].low == 100
    assert narrowed.params["n_estimators"].high == 1000
