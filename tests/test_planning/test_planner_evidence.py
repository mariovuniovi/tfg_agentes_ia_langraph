"""Smoke-tests for planning.validation exports plus plan-exhaustiveness checks.

Full evidence-reference coverage lives in tests/test_planning/test_validation.py.
"""
import pytest

from mlops_agents.contracts.training import RejectedModel, TrainingPlan, TrainingPlanCandidate
from mlops_agents.planning.node import PlannerError
from mlops_agents.planning.validation import (
    PlannerValidationError,
    _check_evidence_references_hybrid,
    _check_plan_exhaustiveness,
)


def test_planning_validation_imports_resolve():
    """Smoke-test: validation module exports are importable."""
    assert PlannerValidationError is not None
    assert _check_plan_exhaustiveness is not None
    assert _check_evidence_references_hybrid is not None


def test_planner_error_importable_from_planning_node():
    """PlannerError is importable from planning.node."""
    assert PlannerError is not None


def test_exhaustiveness_passes_when_all_accounted():
    plan = TrainingPlan(
        problem_type="regression",
        candidates=[TrainingPlanCandidate(priority=1, model_key="ridge", reason="baseline")],
        models_not_recommended=[RejectedModel(model_key="lightgbm_regressor", reason="too slow")],
    )
    _check_plan_exhaustiveness(plan, ["ridge", "lightgbm_regressor"])  # must not raise


def test_exhaustiveness_raises_when_model_missing():
    plan = TrainingPlan(
        problem_type="regression",
        candidates=[TrainingPlanCandidate(priority=1, model_key="ridge", reason="baseline")],
        models_not_recommended=[],
    )
    with pytest.raises(PlannerValidationError, match="lightgbm_regressor"):
        _check_plan_exhaustiveness(plan, ["ridge", "lightgbm_regressor"])
