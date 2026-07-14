"""Tests for build_planner_validation_context (mlops_agents.planning.context)."""
from mlops_agents.models.loader import get_models_for
from mlops_agents.planning.context import build_planner_validation_context


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


def test_context_returns_experiences_and_models():
    """build_planner_validation_context always returns model keys; experiences depend on DB state."""
    ctx = build_planner_validation_context(_regression_profile(), {"target_column": "y"}, "regression")
    # available_model_keys is always populated from the registry
    assert len(ctx.available_model_keys) > 0
    # similar_experiences is a list (may be empty if DB has no data yet)
    assert isinstance(ctx.similar_experiences, list)


def test_context_available_models_regression_only():
    ctx = build_planner_validation_context(_regression_profile(), {}, "regression")
    registry_keys = {m.model_key for m in get_models_for("regression")}
    assert set(ctx.available_model_keys) == registry_keys


def test_context_available_models_classification_only():
    ctx = build_planner_validation_context(_classification_profile(), {}, "classification")
    registry_keys = {m.model_key for m in get_models_for("classification")}
    assert set(ctx.available_model_keys) == registry_keys


def test_context_available_models_forecasting_only():
    ctx = build_planner_validation_context(_forecasting_profile(), {}, "forecasting")
    registry_keys = {m.model_key for m in get_models_for("forecasting")}
    assert set(ctx.available_model_keys) == registry_keys


def test_context_matched_rules_have_rule_id_and_summary():
    ctx = build_planner_validation_context(_regression_profile(), {}, "regression")
    for r in ctx.matched_rules:
        assert "rule_id" in r
        assert "summary" in r
