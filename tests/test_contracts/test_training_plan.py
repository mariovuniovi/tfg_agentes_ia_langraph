"""Unit tests for cross-cutting training contracts."""

import pytest
from pydantic import ValidationError

from mlops_agents.contracts.training import (
    RejectedModel,
    SearchParamOverride,
    TrainingPlan,
    TrainingPlanCandidate,
    TrialBudget,
)


def test_search_param_override_low_high_only():
    o = SearchParamOverride(low=0.01, high=0.1)
    assert o.low == 0.01 and o.high == 0.1 and o.choices is None


def test_search_param_override_choices_only():
    o = SearchParamOverride(choices=[300, 500, 800])
    assert o.choices == [300, 500, 800] and o.low is None and o.high is None


def test_search_param_override_rejects_both():
    with pytest.raises(ValidationError, match="exactly one"):
        SearchParamOverride(low=0.01, high=0.1, choices=[0.05])


def test_search_param_override_rejects_neither():
    with pytest.raises(ValidationError, match="exactly one"):
        SearchParamOverride()


def test_search_param_override_rejects_inverted_range():
    with pytest.raises(ValidationError, match="low.*<= high|<= high"):
        SearchParamOverride(low=0.5, high=0.1)


def test_training_plan_candidate_minimal():
    c = TrainingPlanCandidate(priority=1, model_key="lightgbm_classifier")
    assert c.priority == 1
    assert c.model_key == "lightgbm_classifier"
    assert c.search_space_override is None
    assert c.reason == ""


def test_training_plan_unique_priorities_required():
    candidates = [
        TrainingPlanCandidate(priority=1, model_key="logistic_regression"),
        TrainingPlanCandidate(priority=1, model_key="lightgbm_classifier"),  # duplicate priority
    ]
    with pytest.raises(ValidationError, match="unique"):
        TrainingPlan(problem_type="classification", candidates=candidates)


def test_training_plan_default_trial_budget():
    plan = TrainingPlan(
        problem_type="classification",
        candidates=[TrainingPlanCandidate(priority=1, model_key="logistic_regression")],
    )
    assert plan.trial_budget.total_trials == 60
    assert plan.trial_budget.allocation_strategy == "priority_weighted"


def test_trial_budget_field_default_factory():
    """Two TrialBudget instances must be independent (no shared mutable default)."""
    a = TrialBudget()
    b = TrialBudget()
    assert a is not b


def test_rejected_model_basic():
    r = RejectedModel(model_key="lstm", reason="too small")
    assert r.model_key == "lstm" and r.reason == "too small"
