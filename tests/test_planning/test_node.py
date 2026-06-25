"""Tests for planner_node orchestration.

Validation checks have their own dedicated test suites — these tests only verify that
planner_node orchestrates correctly (build context → run agent → assemble Command).
All four validation functions are patched so we test ONLY the node's plumbing.
"""
from unittest.mock import MagicMock, patch

import pytest

from mlops_agents.contracts.planner import (
    CandidateSpec,
    DecisionBasis,
    EvidenceReference,
    PlannerOutput,
    RejectedModelSpec,
)
from mlops_agents.contracts.training import TrainingPlan
from mlops_agents.planning.node import planner_node

# All forecasting model keys from registry (minus "ets" which is the candidate)
_FORECASTING_REJECTED_KEYS = [
    "naive",
    "seasonal_naive",
    "auto_arima",
    "random_forest_forecaster",
    "extra_trees_forecaster",
    "gbm_forecaster",
    "lightgbm_forecaster",
    "xgboost_forecaster",
    "svr_forecaster",
]


def _make_output_for(problem_type: str = "forecasting") -> PlannerOutput:
    """Build a minimal valid PlannerOutput for the given problem type.

    All validation checks are patched in tests that use this, so the plan doesn't
    need to pass the validation chain — it just needs to be a valid PlannerOutput object.
    """
    ets_ref = EvidenceReference(source="registry", source_id="ets")
    rejected = [
        RejectedModelSpec(
            model_key=k,
            reason="not suitable for this dataset profile",
            evidence_refs=[EvidenceReference(source="registry", source_id=k)],
        )
        for k in _FORECASTING_REJECTED_KEYS
    ]
    return PlannerOutput(
        planning_analysis="ok",
        decision_basis=DecisionBasis(
            primary_evidence=[EvidenceReference(source="dataset_profile", source_id=None)],
            secondary_evidence=[],
            final_strategy="prefer ets",
        ),
        evidence_used=[],
        evidence_conflicts=[],
        risks_or_warnings=[],
        plan=TrainingPlan(
            problem_type=problem_type,
            candidates=[
                CandidateSpec(model_key="ets", priority=1, reason="ok", evidence_refs=[ets_ref])
            ],
            models_not_recommended=rejected,
        ),
    )


@patch("mlops_agents.planning.node._check_conflict_resolution_present_if_flagged")
@patch("mlops_agents.planning.node._check_evidence_references_hybrid")
@patch("mlops_agents.planning.node._check_plan_exhaustiveness")
@patch("mlops_agents.planning.node._check_plan_integrity")
@patch("mlops_agents.planning.node.build_planner_validation_context")
@patch("mlops_agents.planning.node.build_planner_agent")
@patch("mlops_agents.planning.node.build_planner_tools")
@patch("mlops_agents.planning.node.build_dataset_profile")
def test_planner_node_happy_path(
    mock_profile,
    mock_build_tools,
    mock_build_agent,
    mock_build_ctx,
    mock_integrity,
    mock_exhaust,
    mock_evidence,
    mock_conflict,
    tmp_path,
):
    """Validation has its own dedicated tests — patch the checks here so we test ONLY
    the node's orchestration (build context → run agent → assemble Command). This is
    cleaner than rebinding ToolTrace in the node module, which is brittle and obscures intent."""
    # DatasetProfile has required fields — use a MagicMock to avoid field validation
    mock_profile_instance = MagicMock()
    mock_profile_instance.model_dump.return_value = {
        "schema_version": 1,
        "problem_type": "forecasting",
        "n_rows": "small",
        "n_features": "small",
        "missing_rate": "none",
        "n_categorical_features": "none",
        "n_numerical_features": "few",
    }
    mock_profile.return_value = mock_profile_instance

    mock_ctx = MagicMock()
    mock_ctx.problem_type = "forecasting"
    mock_ctx.task_metadata = {}
    mock_ctx.available_model_keys = ["ets"]
    mock_ctx.similar_experiences = []
    mock_ctx.matched_rules = []
    mock_ctx.rules_by_id = {}
    mock_build_ctx.return_value = mock_ctx
    mock_build_tools.return_value = []

    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {
        "structured_response": _make_output_for("forecasting"),
        "messages": [],
    }
    mock_build_agent.return_value = fake_agent

    # 35 rows + forecast_horizon=1 satisfies capacity check (k_max=4)
    rows = "\n".join(f"2024-01-{i + 1:02d},{ i + 1}" for i in range(35))
    csv = tmp_path / "p.csv"
    csv.write_text("ds,y\n" + rows + "\n")
    state = {
        "processed_dataset_path": str(csv),
        "problem_type": "forecasting",
        "task_metadata": {"target_column": "y", "datetime_column": "ds", "forecast_horizon": 1},
    }
    result = planner_node(state)

    assert result.goto == "workflow_controller"
    assert "training_plan" in result.update
    assert "planner_tool_trace" in result.update
    # All four validation checks were invoked exactly once
    mock_integrity.assert_called_once()
    mock_exhaust.assert_called_once()
    mock_evidence.assert_called_once()
    mock_conflict.assert_called_once()


@patch("mlops_agents.planning.node._check_conflict_resolution_present_if_flagged")
@patch("mlops_agents.planning.node._check_evidence_references_hybrid")
@patch("mlops_agents.planning.node._check_plan_exhaustiveness")
@patch("mlops_agents.planning.node._check_plan_integrity")
@patch("mlops_agents.planning.node.build_planner_validation_context")
@patch("mlops_agents.planning.node.build_planner_agent")
@patch("mlops_agents.planning.node.build_planner_tools")
@patch("mlops_agents.planning.node.build_dataset_profile")
def test_planner_node_status_ok_no_retry(
    mock_profile,
    mock_build_tools,
    mock_build_agent,
    mock_build_ctx,
    mock_integrity,
    mock_exhaust,
    mock_evidence,
    mock_conflict,
    tmp_path,
):
    """planner_status is 'ok' and planner_retry_used is False on first-attempt success."""
    mock_profile_instance = MagicMock()
    mock_profile_instance.model_dump.return_value = {}
    mock_profile.return_value = mock_profile_instance

    mock_ctx = MagicMock()
    mock_ctx.problem_type = "forecasting"
    mock_ctx.task_metadata = {}
    mock_ctx.available_model_keys = ["ets"]
    mock_ctx.similar_experiences = []
    mock_ctx.matched_rules = []
    mock_ctx.rules_by_id = {}
    mock_build_ctx.return_value = mock_ctx
    mock_build_tools.return_value = []

    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {
        "structured_response": _make_output_for("forecasting"),
        "messages": [],
    }
    mock_build_agent.return_value = fake_agent

    # 35 rows + forecast_horizon=1 satisfies capacity check (k_max=4)
    rows = "\n".join(f"2024-01-{i + 1:02d},{i + 1}" for i in range(35))
    csv = tmp_path / "p.csv"
    csv.write_text("ds,y\n" + rows + "\n")
    state = {
        "processed_dataset_path": str(csv),
        "problem_type": "forecasting",
        "task_metadata": {"forecast_horizon": 1},
    }
    result = planner_node(state)

    assert result.update["planner_status"] == "ok"
    assert result.update["planner_retry_used"] is False


@patch("mlops_agents.planning.node._check_conflict_resolution_present_if_flagged")
@patch("mlops_agents.planning.node._check_evidence_references_hybrid")
@patch("mlops_agents.planning.node._check_plan_exhaustiveness")
@patch("mlops_agents.planning.node._check_plan_integrity")
@patch("mlops_agents.planning.node.build_planner_validation_context")
@patch("mlops_agents.planning.node.build_planner_agent")
@patch("mlops_agents.planning.node.build_planner_tools")
@patch("mlops_agents.planning.node.build_dataset_profile")
def test_planner_node_retry_on_first_failure(
    mock_profile,
    mock_build_tools,
    mock_build_agent,
    mock_build_ctx,
    mock_integrity,
    mock_exhaust,
    mock_evidence,
    mock_conflict,
    tmp_path,
):
    """When first attempt fails validation, node retries once and succeeds."""
    from mlops_agents.planning.validation import PlannerValidationError

    mock_profile_instance = MagicMock()
    mock_profile_instance.model_dump.return_value = {}
    mock_profile.return_value = mock_profile_instance

    mock_ctx = MagicMock()
    mock_ctx.problem_type = "forecasting"
    mock_ctx.task_metadata = {}
    mock_ctx.available_model_keys = ["ets"]
    mock_ctx.similar_experiences = []
    mock_ctx.matched_rules = []
    mock_ctx.rules_by_id = {}
    mock_build_ctx.return_value = mock_ctx
    mock_build_tools.return_value = []

    good_output = _make_output_for("forecasting")

    call_count = 0

    def fake_invoke(inputs, config=None):
        nonlocal call_count
        call_count += 1
        return {"structured_response": good_output, "messages": []}

    fake_agent = MagicMock()
    fake_agent.invoke.side_effect = fake_invoke
    mock_build_agent.return_value = fake_agent

    # First call to _check_plan_integrity raises; second succeeds
    mock_integrity.side_effect = [
        PlannerValidationError("integrity fail on attempt 1"),
        None,
    ]

    # 35 rows + forecast_horizon=1 satisfies capacity check (k_max=4)
    rows = "\n".join(f"2024-01-{i + 1:02d},{i + 1}" for i in range(35))
    csv = tmp_path / "p.csv"
    csv.write_text("ds,y\n" + rows + "\n")
    state = {
        "processed_dataset_path": str(csv),
        "problem_type": "forecasting",
        "task_metadata": {"forecast_horizon": 1},
    }
    result = planner_node(state)

    assert result.update["planner_status"] == "retry_ok"
    assert result.update["planner_retry_used"] is True
    assert call_count == 2


@patch("mlops_agents.planning.node._check_conflict_resolution_present_if_flagged")
@patch("mlops_agents.planning.node._check_evidence_references_hybrid")
@patch("mlops_agents.planning.node._check_plan_exhaustiveness")
@patch("mlops_agents.planning.node._check_plan_integrity")
@patch("mlops_agents.planning.node.build_planner_validation_context")
@patch("mlops_agents.planning.node.build_planner_agent")
@patch("mlops_agents.planning.node.build_planner_tools")
@patch("mlops_agents.planning.node.build_dataset_profile")
def test_planner_node_raises_after_two_failures(
    mock_profile,
    mock_build_tools,
    mock_build_agent,
    mock_build_ctx,
    mock_integrity,
    mock_exhaust,
    mock_evidence,
    mock_conflict,
    tmp_path,
):
    """When both attempts fail validation, PlannerError is raised."""
    from mlops_agents.planning.node import PlannerError
    from mlops_agents.planning.validation import PlannerValidationError

    mock_profile_instance = MagicMock()
    mock_profile_instance.model_dump.return_value = {}
    mock_profile.return_value = mock_profile_instance

    mock_ctx = MagicMock()
    mock_ctx.problem_type = "forecasting"
    mock_ctx.task_metadata = {}
    mock_ctx.available_model_keys = ["ets"]
    mock_ctx.similar_experiences = []
    mock_ctx.matched_rules = []
    mock_ctx.rules_by_id = {}
    mock_build_ctx.return_value = mock_ctx
    mock_build_tools.return_value = []

    good_output = _make_output_for("forecasting")
    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {"structured_response": good_output, "messages": []}
    mock_build_agent.return_value = fake_agent

    # Both calls to _check_plan_integrity raise
    mock_integrity.side_effect = PlannerValidationError("always fails")

    # 35 rows + forecast_horizon=1 satisfies capacity check (k_max=4)
    rows = "\n".join(f"2024-01-{i + 1:02d},{i + 1}" for i in range(35))
    csv = tmp_path / "p.csv"
    csv.write_text("ds,y\n" + rows + "\n")
    state = {
        "processed_dataset_path": str(csv),
        "problem_type": "forecasting",
        "task_metadata": {"forecast_horizon": 1},
    }
    with pytest.raises(PlannerError):
        planner_node(state)


@patch("mlops_agents.planning.node._check_conflict_resolution_present_if_flagged")
@patch("mlops_agents.planning.node._check_evidence_references_hybrid")
@patch("mlops_agents.planning.node._check_plan_exhaustiveness")
@patch("mlops_agents.planning.node._check_plan_integrity")
@patch("mlops_agents.planning.node.build_planner_validation_context")
@patch("mlops_agents.planning.node.build_planner_agent")
@patch("mlops_agents.planning.node.build_planner_tools")
@patch("mlops_agents.planning.node.build_dataset_profile")
def test_planner_node_candidates_sorted_by_priority(
    mock_profile,
    mock_build_tools,
    mock_build_agent,
    mock_build_ctx,
    mock_integrity,
    mock_exhaust,
    mock_evidence,
    mock_conflict,
    tmp_path,
):
    """Candidates in the returned training_plan dict are sorted ascending by priority."""
    mock_profile_instance = MagicMock()
    mock_profile_instance.model_dump.return_value = {}
    mock_profile.return_value = mock_profile_instance

    mock_ctx = MagicMock()
    mock_ctx.problem_type = "forecasting"
    mock_ctx.task_metadata = {}
    mock_ctx.available_model_keys = ["ets", "naive"]
    mock_ctx.similar_experiences = []
    mock_ctx.matched_rules = []
    mock_ctx.rules_by_id = {}
    mock_build_ctx.return_value = mock_ctx
    mock_build_tools.return_value = []

    ets_ref = EvidenceReference(source="registry", source_id="ets")
    naive_ref = EvidenceReference(source="registry", source_id="naive")
    rejected = [
        RejectedModelSpec(
            model_key=k,
            reason="not suitable",
            evidence_refs=[EvidenceReference(source="registry", source_id=k)],
        )
        for k in _FORECASTING_REJECTED_KEYS
        if k not in ("ets", "naive")
    ]
    output_with_reversed_priorities = PlannerOutput(
        planning_analysis="ok",
        decision_basis=DecisionBasis(
            primary_evidence=[EvidenceReference(source="dataset_profile", source_id=None)],
            secondary_evidence=[],
            final_strategy="prefer ets",
        ),
        evidence_used=[],
        evidence_conflicts=[],
        risks_or_warnings=[],
        plan=TrainingPlan(
            problem_type="forecasting",
            candidates=[
                CandidateSpec(model_key="naive", priority=2, reason="ok", evidence_refs=[naive_ref]),
                CandidateSpec(model_key="ets", priority=1, reason="ok", evidence_refs=[ets_ref]),
            ],
            models_not_recommended=rejected,
        ),
    )

    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {
        "structured_response": output_with_reversed_priorities,
        "messages": [],
    }
    mock_build_agent.return_value = fake_agent

    # 35 rows + forecast_horizon=1 satisfies capacity check (k_max=4)
    rows = "\n".join(f"2024-01-{i + 1:02d},{i + 1}" for i in range(35))
    csv = tmp_path / "p.csv"
    csv.write_text("ds,y\n" + rows + "\n")
    state = {
        "processed_dataset_path": str(csv),
        "problem_type": "forecasting",
        "task_metadata": {"forecast_horizon": 1},
    }
    result = planner_node(state)

    candidates = result.update["training_plan"]["candidates"]
    priorities = [c["priority"] for c in candidates]
    assert priorities == sorted(priorities), f"Expected sorted priorities, got {priorities}"
    assert candidates[0]["model_key"] == "ets"


@patch("mlops_agents.planning.node._check_conflict_resolution_present_if_flagged")
@patch("mlops_agents.planning.node._check_evidence_references_hybrid")
@patch("mlops_agents.planning.node._check_plan_exhaustiveness")
@patch("mlops_agents.planning.node._check_plan_integrity")
@patch("mlops_agents.planning.node.build_planner_validation_context")
@patch("mlops_agents.planning.node.build_planner_agent")
@patch("mlops_agents.planning.node.build_planner_tools")
@patch("mlops_agents.planning.node.build_dataset_profile")
def test_planner_node_injects_policy_forecasting_settings(
    mock_profile, mock_build_tools, mock_build_agent, mock_build_ctx,
    mock_integrity, mock_exhaust, mock_evidence, mock_conflict, tmp_path,
):
    """The plan's forecasting_settings must equal the deterministic policy output,
    regardless of what the LLM returns (here: forecasting_settings=None)."""
    mock_profile_instance = MagicMock()
    mock_profile_instance.model_dump.return_value = {}
    mock_profile.return_value = mock_profile_instance
    mock_ctx = MagicMock()
    mock_ctx.problem_type = "forecasting"
    mock_ctx.task_metadata = {}
    mock_ctx.available_model_keys = ["ets"]
    mock_ctx.similar_experiences = []
    mock_ctx.matched_rules = []
    mock_ctx.rules_by_id = {}
    mock_build_ctx.return_value = mock_ctx
    mock_build_tools.return_value = []
    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {
        "structured_response": _make_output_for("forecasting"),  # plan.forecasting_settings is None
        "messages": [],
    }
    mock_build_agent.return_value = fake_agent

    # 60 observations, horizon 8 -> capacity policy => expanding_window, 2 folds (NOT single_split)
    rows = "\n".join(f"2023-{(i % 12) + 1:02d}-01,{i}" for i in range(60))
    csv = tmp_path / "p.csv"
    csv.write_text("ds,y\n" + rows + "\n")
    state = {
        "processed_dataset_path": str(csv),
        "problem_type": "forecasting",
        "task_metadata": {
            "target_column": "y", "datetime_column": "ds",
            "forecast_horizon": 8, "exogenous_columns": [],
        },
    }
    result = planner_node(state)
    fs = result.update["training_plan"]["forecasting_settings"]
    assert fs is not None
    assert fs["validation_strategy"]["type"] == "expanding_window"
    assert fs["validation_strategy"]["n_folds"] == 2


@patch("mlops_agents.planning.node._check_conflict_resolution_present_if_flagged")
@patch("mlops_agents.planning.node._check_evidence_references_hybrid")
@patch("mlops_agents.planning.node._check_plan_exhaustiveness")
@patch("mlops_agents.planning.node._check_plan_integrity")
@patch("mlops_agents.planning.node.build_planner_validation_context")
@patch("mlops_agents.planning.node.build_planner_agent")
@patch("mlops_agents.planning.node.build_planner_tools")
@patch("mlops_agents.planning.node.build_dataset_profile")
def test_planner_node_emits_executor_reconstructable_training_plan(
    mock_profile, mock_build_tools, mock_build_agent, mock_build_ctx,
    mock_integrity, mock_exhaust, mock_evidence, mock_conflict, tmp_path,
):
    """The state's training_plan must be a full, executor-ready TrainingPlan:
    rebuildable via TrainingPlan.model_validate with the code-resolved settings."""
    from mlops_agents.contracts.training import TrainingPlan

    mock_profile_instance = MagicMock()
    mock_profile_instance.model_dump.return_value = {}
    mock_profile.return_value = mock_profile_instance
    mock_ctx = MagicMock()
    mock_ctx.problem_type = "forecasting"
    mock_ctx.task_metadata = {}
    mock_ctx.available_model_keys = ["ets"]
    mock_ctx.similar_experiences = []
    mock_ctx.matched_rules = []
    mock_ctx.rules_by_id = {}
    mock_build_ctx.return_value = mock_ctx
    mock_build_tools.return_value = []
    fake_agent = MagicMock()
    fake_agent.invoke.return_value = {
        "structured_response": _make_output_for("forecasting"),
        "messages": [],
    }
    mock_build_agent.return_value = fake_agent

    rows = "\n".join(f"2023-{(i % 12) + 1:02d}-01,{i}" for i in range(60))
    csv = tmp_path / "p.csv"
    csv.write_text("ds,y\n" + rows + "\n")
    state = {
        "processed_dataset_path": str(csv),
        "problem_type": "forecasting",
        "task_metadata": {
            "target_column": "y", "datetime_column": "ds",
            "forecast_horizon": 8, "exogenous_columns": [],
        },
    }
    result = planner_node(state)

    rebuilt = TrainingPlan.model_validate(result.update["training_plan"])
    assert rebuilt.forecasting_settings is not None
    assert rebuilt.forecasting_settings.validation_strategy.type == "expanding_window"
    assert [c.model_key for c in rebuilt.candidates] == ["ets"]


@patch("mlops_agents.planning.node.build_planner_validation_context")
@patch("mlops_agents.planning.node.build_planner_agent")
@patch("mlops_agents.planning.node.build_planner_tools")
@patch("mlops_agents.planning.node.build_dataset_profile")
def test_planner_node_raises_plannererror_on_too_small_forecasting(
    mock_profile, mock_build_tools, mock_build_agent, mock_build_ctx, tmp_path,
):
    import pytest

    from mlops_agents.planning.node import PlannerError
    mock_profile_instance = MagicMock()
    mock_profile_instance.model_dump.return_value = {}
    mock_profile.return_value = mock_profile_instance
    mock_build_ctx.return_value = MagicMock()
    mock_build_tools.return_value = []
    # 10 rows, horizon 8 -> capacity check fails (need >= 46)
    rows = "\n".join(f"2023-{(i % 12) + 1:02d}-01,{i}" for i in range(10))
    csv = tmp_path / "tiny.csv"
    csv.write_text("ds,y\n" + rows + "\n")
    state = {
        "processed_dataset_path": str(csv),
        "problem_type": "forecasting",
        "task_metadata": {"target_column": "y", "datetime_column": "ds",
                          "forecast_horizon": 8, "exogenous_columns": []},
    }
    with pytest.raises(PlannerError, match="capacity check failed"):
        planner_node(state)
