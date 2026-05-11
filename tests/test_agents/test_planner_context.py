import pytest
from pathlib import Path
from mlops_agents.agents.planner import build_planner_context, PlannerError
from mlops_agents.experience.pool import ExperiencePool
from mlops_agents.models.loader import get_models_for


@pytest.fixture()
def empty_pool(tmp_path: Path) -> ExperiencePool:
    return ExperiencePool(tmp_path / "test.db")


def _regression_profile() -> dict:
    return {
        "schema_version": 1, "problem_type": "regression",
        "n_rows": "medium", "n_features": "medium",
        "missing_rate": "none", "n_categorical_features": "none",
        "n_numerical_features": "many",
        "target_distribution": "near_normal",
    }


def _classification_profile() -> dict:
    return {
        "schema_version": 1, "problem_type": "classification",
        "n_rows": "medium", "n_features": "medium",
        "missing_rate": "none", "n_categorical_features": "none",
        "n_numerical_features": "many",
        "n_classes": "binary", "class_balance": "balanced",
    }


def _forecasting_profile() -> dict:
    return {
        "schema_version": 1, "problem_type": "forecasting",
        "n_rows": "medium", "n_features": "small",
        "missing_rate": "none", "n_categorical_features": "none",
        "n_numerical_features": "few",
        "n_series": "single", "history_length": "long",
        "horizon_difficulty": "short", "seasonality_detected": False,
    }


def test_context_empty_pool_returns_no_experiences(empty_pool):
    ctx = build_planner_context(_regression_profile(), {"target_column": "y"}, "regression", empty_pool)
    assert ctx.similar_experiences == []
    assert len(ctx.available_models) > 0


def test_context_available_models_regression_only(empty_pool):
    ctx = build_planner_context(_regression_profile(), {}, "regression", empty_pool)
    registry_keys = {m.model_key for m in get_models_for("regression")}
    assert set(ctx.available_models) == registry_keys


def test_context_available_models_classification_only(empty_pool):
    ctx = build_planner_context(_classification_profile(), {}, "classification", empty_pool)
    registry_keys = {m.model_key for m in get_models_for("classification")}
    assert set(ctx.available_models) == registry_keys


def test_context_available_models_forecasting_only(empty_pool):
    ctx = build_planner_context(_forecasting_profile(), {}, "forecasting", empty_pool)
    registry_keys = {m.model_key for m in get_models_for("forecasting")}
    assert set(ctx.available_models) == registry_keys


def test_context_matched_rules_have_rule_id_and_summary(empty_pool):
    ctx = build_planner_context(_regression_profile(), {}, "regression", empty_pool)
    for r in ctx.matched_rules:
        assert "rule_id" in r
        assert "summary" in r
