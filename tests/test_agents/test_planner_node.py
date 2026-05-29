"""Tests for planner_node — LLM call + retry loop (SP5)."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mlops_agents.agents.planner import PlannerError, planner_node
from mlops_agents.contracts.planner import PlannerOutput
from mlops_agents.contracts.training import TrainingPlan, TrainingPlanCandidate
from mlops_agents.models.loader import get_models_for


def _full_regression_plan() -> TrainingPlan:
    """Plan that includes ALL regression models (passes exhaustiveness check)."""
    models = get_models_for("regression")
    candidates = [
        TrainingPlanCandidate(priority=i + 1, model_key=m.model_key, reason="test")
        for i, m in enumerate(models)
    ]
    return TrainingPlan(problem_type="regression", candidates=candidates)


def _valid_planner_output() -> PlannerOutput:
    return PlannerOutput(
        planning_analysis="Selected all regression models for test.",
        plan=_full_regression_plan(),
    )


def _minimal_state(tmp_path: Path) -> dict:
    import pandas as pd

    csv = tmp_path / "data.csv"
    pd.DataFrame({"f1": range(200), "target": range(200)}).to_csv(csv, index=False)
    return {
        "processed_dataset_path": str(csv),
        "problem_type": "regression",
        "task_metadata": {"target_column": "target"},
        "messages": [],
        "planner_analysis": None,
        "planner_evidence_used": [],
        "planner_warnings": [],
        "planner_status": None,
        "planner_retry_used": None,
        "training_plan": None,
        "error_message": "",
    }


def test_planner_node_happy_path(tmp_path):
    state = _minimal_state(tmp_path)
    mock_output = _valid_planner_output()

    with patch("mlops_agents.agents.registry.get_agent") as mock_get_agent, \
         patch("mlops_agents.agents.planner.ExperiencePool") as mock_pool_cls:
        mock_pool_cls.return_value.find_similar.return_value = []
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = mock_output
        mock_get_agent.return_value = mock_llm

        cmd = planner_node(state)

    update = cmd.update
    assert update["planner_status"] == "ok"
    assert update["planner_retry_used"] is False
    assert update["planner_analysis"] == "Selected all regression models for test."
    assert update["training_plan"] is not None


def test_planner_node_retry_on_first_failure(tmp_path):
    state = _minimal_state(tmp_path)
    mock_output = _valid_planner_output()

    with patch("mlops_agents.agents.registry.get_agent") as mock_get_agent, \
         patch("mlops_agents.agents.planner.ExperiencePool") as mock_pool_cls:
        mock_pool_cls.return_value.find_similar.return_value = []
        mock_llm = MagicMock()
        invoke_mock = mock_llm.invoke
        invoke_mock.side_effect = [PlannerError("bad evidence"), mock_output]
        mock_get_agent.return_value = mock_llm

        cmd = planner_node(state)

    update = cmd.update
    assert update["planner_status"] == "retry_ok"
    assert update["planner_retry_used"] is True
    assert invoke_mock.call_count == 2


def test_planner_node_raises_after_two_failures(tmp_path):
    state = _minimal_state(tmp_path)

    with patch("mlops_agents.agents.registry.get_agent") as mock_get_agent, \
         patch("mlops_agents.agents.planner.ExperiencePool") as mock_pool_cls:
        mock_pool_cls.return_value.find_similar.return_value = []
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = PlannerError("bad")
        mock_get_agent.return_value = mock_llm

        with pytest.raises(PlannerError, match="Planner failed after retry"):
            planner_node(state)


def test_planner_node_check_integrity_failure_triggers_retry(tmp_path):
    """_check_plan_exhaustiveness failure also triggers retry."""
    state = _minimal_state(tmp_path)
    models = get_models_for("regression")
    # Plan missing models_not_recommended for some models (exhaustiveness fails)
    bad_plan = TrainingPlan(
        problem_type="regression",
        candidates=[TrainingPlanCandidate(priority=1, model_key=models[0].model_key, reason="x")],
        models_not_recommended=[],
    )
    bad_output = PlannerOutput(planning_analysis="bad", plan=bad_plan)
    good_output = _valid_planner_output()

    with patch("mlops_agents.agents.registry.get_agent") as mock_get_agent, \
         patch("mlops_agents.agents.planner.ExperiencePool") as mock_pool_cls:
        mock_pool_cls.return_value.find_similar.return_value = []
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [bad_output, good_output]
        mock_get_agent.return_value = mock_llm

        cmd = planner_node(state)

    assert cmd.update["planner_status"] == "retry_ok"
