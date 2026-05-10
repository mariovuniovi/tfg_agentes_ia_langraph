"""Tests for ML rules reader."""
import pytest
from pydantic import ValidationError
from mlops_agents.knowledge.reader import MLRule, match_rules


def _cls_profile(n_rows: str = "medium", class_balance: str = "balanced") -> dict:
    return {
        "problem_type": "classification",
        "n_rows": n_rows, "n_features": "small",
        "missing_rate": "none", "n_categorical_features": "none",
        "n_numerical_features": "few",
        "n_classes": "binary", "class_balance": class_balance,
    }


def _fc_profile(history_length: str = "short", seasonality_detected: bool = False) -> dict:
    return {
        "problem_type": "forecasting",
        "n_rows": "medium", "n_features": "small",
        "missing_rate": "none", "n_categorical_features": "none",
        "n_numerical_features": "few",
        "n_series": "single", "history_length": history_length,
        "frequency": "MS", "horizon_difficulty": "short",
        "exogenous_features_available": False,
        "seasonality_detected": seasonality_detected,
        "trend_detected": False, "stationarity": True,
    }


def test_match_rules_returns_matching_rules():
    rules = match_rules(_cls_profile(n_rows="very_small"))
    assert "classification_very_small_prefers_simple_models" in {r.rule_id for r in rules}


def test_match_rules_does_not_return_non_matching():
    rules = match_rules(_cls_profile(n_rows="medium"))
    assert "classification_very_small_prefers_simple_models" not in {r.rule_id for r in rules}


def test_match_rules_forecasting_short_history():
    rules = match_rules(_fc_profile(history_length="short"))
    assert "forecasting_short_history_prefers_statistical" in {r.rule_id for r in rules}


def test_match_rules_list_applies_when():
    """A rule with applies_when value as a list matches if profile value is in the list."""
    rules = match_rules(_fc_profile(history_length="very_short"))
    assert "forecasting_short_history_prefers_statistical" in {r.rule_id for r in rules}


def test_mlrule_rejects_unknown_profile_field():
    with pytest.raises(ValidationError, match="unknown profile field"):
        MLRule(
            rule_id="bad_rule",
            applies_when={"history_lenght": "short"},  # typo
            prefer=["naive"],
            reason="test",
        )


def test_mlrule_rejects_wrong_problem_type_model():
    with pytest.raises(ValidationError, match="does not match"):
        MLRule(
            rule_id="bad_mix",
            applies_when={"problem_type": "forecasting"},
            prefer=["logistic_regression"],  # classification model!
            reason="test",
        )
