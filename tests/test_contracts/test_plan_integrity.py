"""Structural integrity checks for TrainingPlan: overlap, empty reasons,
unknown keys, wrong problem_type, valid plan accepted."""
import pytest
from pydantic import ValidationError

from mlops_agents.contracts.training import (
    RejectedModel,
    TrainingPlan,
    TrainingPlanCandidate,
    TrialBudget,
)


def _plan(*, problem_type="forecasting", candidates=None, rejected=None):
    return TrainingPlan(
        problem_type=problem_type,
        candidates=candidates if candidates is not None else [
            TrainingPlanCandidate(priority=1, model_key="naive"),
        ],
        models_not_recommended=rejected or [],
        trial_budget=TrialBudget(
            total_trials=2, allocation_strategy="equal",
            min_trials_per_candidate=1, max_trials_per_candidate=2,
        ),
    )


def test_overlap_between_candidates_and_rejected_raises():
    with pytest.raises(ValidationError, match="both candidates and models_not_recommended"):
        _plan(
            candidates=[TrainingPlanCandidate(priority=1, model_key="naive")],
            rejected=[RejectedModel(model_key="naive", reason="contradictory")],
        )


def test_empty_rejected_reason_raises():
    with pytest.raises(ValidationError, match=r"models_not_recommended\[ets\].reason is empty"):
        _plan(rejected=[RejectedModel(model_key="ets", reason="")])


def test_whitespace_only_rejected_reason_raises():
    with pytest.raises(ValidationError, match="reason is empty"):
        _plan(rejected=[RejectedModel(model_key="ets", reason="   \t\n")])


def test_unknown_model_key_in_candidates_raises():
    with pytest.raises(ValidationError, match="unknown or wrong-problem-type"):
        _plan(candidates=[TrainingPlanCandidate(priority=1, model_key="nonexistent_model")])


def test_wrong_problem_type_model_key_raises():
    # logistic_regression is a classifier; using it in a forecasting plan must fail
    with pytest.raises(ValidationError, match="unknown or wrong-problem-type.*logistic_regression"):
        _plan(candidates=[TrainingPlanCandidate(priority=1, model_key="logistic_regression")])


def test_unknown_model_key_in_rejected_raises():
    with pytest.raises(ValidationError, match="unknown or wrong-problem-type"):
        _plan(rejected=[RejectedModel(model_key="not_a_real_model", reason="just because")])


def test_valid_plan_with_candidates_and_rejected_passes():
    plan = _plan(
        candidates=[
            TrainingPlanCandidate(priority=1, model_key="naive"),
            TrainingPlanCandidate(priority=2, model_key="ets"),
        ],
        rejected=[
            RejectedModel(
                model_key="lightgbm_forecaster",
                reason="dataset has 30 rows; supervised lag-based models overfit at this size",
            ),
        ],
    )
    assert {c.model_key for c in plan.candidates} == {"naive", "ets"}
    assert plan.models_not_recommended[0].model_key == "lightgbm_forecaster"


def test_error_message_is_deterministic_via_sorted():
    # Two overlapping keys → sorted output regardless of insertion order
    with pytest.raises(ValidationError) as exc:
        _plan(
            candidates=[
                TrainingPlanCandidate(priority=1, model_key="ets"),
                TrainingPlanCandidate(priority=2, model_key="naive"),
            ],
            rejected=[
                RejectedModel(model_key="naive", reason="x"),
                RejectedModel(model_key="ets", reason="y"),
            ],
        )
    msg = str(exc.value)
    assert "['ets', 'naive']" in msg  # sorted alphabetically
