"""Tests for default_training_plan."""
import pytest
from mlops_agents.contracts.profile import DatasetProfile
from mlops_agents.training.default_plans import default_training_plan


def _profile(problem_type: str, n_rows: str = "medium", **kwargs: object) -> DatasetProfile:
    """Minimal DatasetProfile for plan eligibility tests."""
    return DatasetProfile(
        problem_type=problem_type,  # type: ignore[arg-type]
        n_rows=n_rows,  # type: ignore[arg-type]
        n_features="medium",
        missing_rate="none",
        n_categorical_features="none",
        n_numerical_features="some",
        **kwargs,
    )


def test_default_classification_plan_lists_all_eligible():
    plan = default_training_plan("classification", _profile("classification"))
    keys = {c.model_key for c in plan.candidates}
    assert "logistic_regression" in keys
    assert "lightgbm_classifier" in keys
    assert plan.problem_type == "classification"


def test_default_regression_plan_lists_all_eligible():
    plan = default_training_plan("regression", _profile("regression"))
    keys = {c.model_key for c in plan.candidates}
    assert "ridge" in keys
    assert "lightgbm_regressor" in keys


def test_default_forecasting_plan_lists_all_eligible():
    plan = default_training_plan("forecasting", _profile("forecasting", history_length="long"))
    keys = {c.model_key for c in plan.candidates}
    assert "auto_arima" in keys
    assert "lightgbm_forecaster" in keys


def test_default_plan_skips_min_rows_violators():
    """very_small → boosters with min_rows=500 excluded."""
    plan = default_training_plan("regression", _profile("regression", n_rows="very_small"))
    keys = {c.model_key for c in plan.candidates}
    assert "ridge" in keys
    assert "lightgbm_regressor" not in keys


def test_default_plan_priorities_unique_and_ordered():
    plan = default_training_plan("classification", _profile("classification"))
    priorities = [c.priority for c in plan.candidates]
    assert len(priorities) == len(set(priorities))
    assert priorities == sorted(priorities)


def test_default_plan_eligible_for_very_small_classification():
    plan = default_training_plan("classification", _profile("classification", n_rows="very_small"))
    assert "logistic_regression" in {c.model_key for c in plan.candidates}
