"""End-to-end integration tests covering the full pipeline.

Three tiers:
1. Deterministic node tests — no LLM, real training (executor_node with real sklearn).
2. Graph-flow tests — mocked LLM + HITL, verify state flows through all nodes.
3. Real-LLM tests — @pytest.mark.integration, require GITHUB_TOKEN + MLflow.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from langchain_core.messages import HumanMessage
from sklearn.datasets import load_iris


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def iris_canonical_csv(tmp_path: Path) -> Path:
    """Canonical iris CSV ready for training (already mapped + validated)."""
    data = load_iris(as_frame=True)
    df = pd.concat([data.data, data.target.rename("target")], axis=1)
    p = tmp_path / "iris_canonical.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture()
def iris_schema_json() -> str:
    return json.dumps({
        "problem_type": "classification",
        "target_column": "target",
        "columns": [
            {"name": "sepal length (cm)", "dtype": "float", "required": True, "nullable": False},
            {"name": "sepal width (cm)",  "dtype": "float", "required": True, "nullable": False},
            {"name": "petal length (cm)", "dtype": "float", "required": True, "nullable": False},
            {"name": "petal width (cm)",  "dtype": "float", "required": True, "nullable": False},
            {"name": "target",            "dtype": "int",   "required": True, "nullable": False},
        ],
    })


@pytest.fixture()
def iris_training_plan() -> dict:
    """Minimal valid TrainingPlan dict for iris classification."""
    return {
        "problem_type": "classification",
        "metric_to_optimize": None,
        "candidates": [
            {"priority": 1, "model_key": "logistic_regression",
             "initial_hyperparameters": {}, "search_space_override": None,
             "requested_trials": None, "reason": ""},
            {"priority": 2, "model_key": "random_forest_classifier",
             "initial_hyperparameters": {}, "search_space_override": None,
             "requested_trials": None, "reason": ""},
        ],
        "models_not_recommended": [],
        "trial_budget": {
            "total_trials": 12,
            "allocation_strategy": "priority_weighted",
            "max_trials_per_candidate": 8,
            "min_trials_per_candidate": 4,
        },
        "forecasting_settings": None,
    }


# ---------------------------------------------------------------------------
# Tier 1: executor_node end-to-end (deterministic, real sklearn, no LLM)
# ---------------------------------------------------------------------------

def test_executor_node_full_training_run(tmp_path, iris_canonical_csv, iris_training_plan, monkeypatch):
    """executor_node must complete training, write artifacts, and return training_run_id."""
    monkeypatch.setattr(
        "mlops_agents.training.executor.settings.experience_pool_dir",
        tmp_path / "pool",
    )
    state = {
        "messages": [],
        "processed_dataset_path": str(iris_canonical_csv),
        "problem_type": "classification",
        "task_metadata": {"target_column": "target"},  # problem_type absent intentionally
        "training_plan": iris_training_plan,
        "_planner_output_record": None,
        "training_run_id": "",
        "trained_model_path": "",
        "training_metrics": {},
        "error_message": "",
    }

    from mlops_agents.graphs.mlops_graph import executor_node
    command = executor_node(state)

    assert command.goto == "supervisor"
    assert command.update["training_run_id"], "MLflow run ID must be set after training"
    assert Path(command.update["trained_model_path"]).exists(), "Champion model file must exist"
    assert command.update["training_metrics"].get("macro_f1", 0) > 0


def test_executor_node_problem_type_not_in_initial_task_metadata(tmp_path, iris_canonical_csv, iris_training_plan, monkeypatch):
    """run_training_plan must receive task_metadata with problem_type even when it is absent from state['task_metadata']."""
    monkeypatch.setattr(
        "mlops_agents.training.executor.settings.experience_pool_dir",
        tmp_path / "pool",
    )
    captured: dict = {}
    original_run = None

    def spy_run(plan, processed_dataset_path, target_column, task_metadata, **kwargs):
        captured["task_metadata"] = dict(task_metadata)
        # delegate to the real implementation
        return original_run(
            plan=plan,
            processed_dataset_path=processed_dataset_path,
            target_column=target_column,
            task_metadata=task_metadata,
            **kwargs,
        )

    import mlops_agents.training.executor as _exec_mod
    original_run = _exec_mod.run_training_plan

    state = {
        "messages": [],
        "processed_dataset_path": str(iris_canonical_csv),
        "problem_type": "classification",
        "task_metadata": {"target_column": "target"},
        "training_plan": iris_training_plan,
        "_planner_output_record": None,
        "training_run_id": "",
        "trained_model_path": "",
        "training_metrics": {},
        "error_message": "",
    }

    with patch.object(_exec_mod, "run_training_plan", side_effect=spy_run):
        from mlops_agents.graphs.mlops_graph import executor_node
        executor_node(state)

    assert captured["task_metadata"].get("problem_type") == "classification"


# ---------------------------------------------------------------------------
# Tier 2: graph-flow test — mocked LLM + HITL auto-approve
# ---------------------------------------------------------------------------

def _make_supervisor_routing_output(next_node: str, reasoning: str = "") -> MagicMock:
    mock = MagicMock()
    mock.next = next_node
    mock.reasoning = reasoning
    return mock


def test_graph_state_flows_through_data_validator_to_supervisor(tmp_path, iris_schema_json, monkeypatch):
    """data_validator_node must populate validation_passed, problem_type, and processed_dataset_path in state."""
    data = load_iris(as_frame=True)
    df = pd.concat([data.data, data.target.rename("target")], axis=1)
    csv = tmp_path / "iris.csv"
    df.to_csv(csv, index=False)

    from langchain_core.messages import ToolMessage

    def make_agent():
        mock = MagicMock()
        tool_msg = ToolMessage(
            content=json.dumps({"output_path": str(csv), "mapped_columns": 5}),
            tool_call_id="t1",
            name="apply_column_mapping",
        )
        schema_msg = ToolMessage(
            content=json.dumps({"passed": True}),
            tool_call_id="t2",
            name="validate_against_schema",
        )
        mock.invoke.return_value = {"messages": [
            tool_msg, schema_msg,
            HumanMessage(content="Validation passed."),
        ]}
        return mock

    state = {
        "messages": [HumanMessage(content="Run pipeline.")],
        "next": "",
        "dataset_paths": [str(csv)],
        "processed_dataset_path": "",
        "schema_json": iris_schema_json,
        "validation_passed": False,
        "validation_report": {},
        "trained_model_path": "",
        "training_run_id": "",
        "training_metrics": {},
        "evaluation_passed": False,
        "evaluation_report": {},
        "best_model_uri": "",
        "deployment_decision": "pending",
        "deployment_status": "",
        "error_message": "",
        "agent_attempt_counts": {"data_validator": 1},
        "dataset_summary": {},
        "problem_type": "",
        "task_metadata": {},
        "training_plan": None,
        "planner_analysis": None,
        "planner_evidence_used": [],
        "planner_warnings": [],
        "planner_status": None,
        "planner_retry_used": None,
        "_planner_output_record": None,
    }

    with patch("mlops_agents.graphs.mlops_graph.get_agent", side_effect=lambda _: make_agent()), \
         patch("mlops_agents.graphs.mlops_graph.interrupt", return_value={"approved": True, "comment": ""}):
        from mlops_agents.graphs.mlops_graph import data_validator_node
        command = data_validator_node(state)

    assert command.goto == "supervisor"
    assert command.update["problem_type"] == "classification"
    assert command.update["validation_passed"] is True
    assert command.update["processed_dataset_path"] != ""
    assert command.update["task_metadata"]["target_column"] == "target"


# ---------------------------------------------------------------------------
# Tier 3: real LLM integration test
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
def test_full_pipeline_iris_classification(tmp_path, iris_schema_json, monkeypatch):
    """Full graph: data_validator → planner → executor on iris. Requires GITHUB_TOKEN + MLflow.

    HITL is auto-approved so this runs non-interactively.
    """
    from mlops_agents.config.constants import GRAPH_RECURSION_LIMIT
    from mlops_agents.graphs.mlops_graph import graph

    monkeypatch.setattr(
        "mlops_agents.training.executor.settings.experience_pool_dir",
        tmp_path / "pool",
    )

    data = load_iris(as_frame=True)
    df = pd.concat([data.data, data.target.rename("target")], axis=1)
    csv = tmp_path / "iris.csv"
    df.to_csv(csv, index=False)

    config = {
        "configurable": {"thread_id": "e2e-iris-classification"},
        "recursion_limit": GRAPH_RECURSION_LIMIT,
    }
    initial_state = {
        "messages": [HumanMessage(content=f"Run the full MLOps pipeline on these raw files: {csv}")],
        "next": "",
        "dataset_paths": [str(csv)],
        "processed_dataset_path": "",
        "schema_json": iris_schema_json,
        "dataset_summary": {},
        "problem_type": "",
        "task_metadata": {},
        "validation_passed": False,
        "validation_report": {},
        "trained_model_path": "",
        "training_run_id": "",
        "training_metrics": {},
        "training_plan": None,
        "planner_analysis": None,
        "planner_evidence_used": [],
        "planner_warnings": [],
        "planner_status": None,
        "planner_retry_used": None,
        "_planner_output_record": None,
        "evaluation_passed": False,
        "evaluation_report": {},
        "best_model_uri": "",
        "deployment_decision": "pending",
        "deployment_status": "",
        "error_message": "",
        "agent_attempt_counts": {},
    }

    # Auto-approve the data validation HITL so the test runs non-interactively
    with patch("mlops_agents.graphs.mlops_graph.interrupt", return_value={"approved": True, "comment": ""}):
        result = graph.invoke(initial_state, config=config)

    assert result is not None
    assert result.get("validation_passed") is True, "Pipeline must pass data validation"
    assert result.get("planner_status") in ("ok", "retry_ok"), "Planner must succeed"
    assert result.get("training_run_id"), "Executor must complete and set training_run_id"
    assert result.get("training_metrics", {}).get("macro_f1", 0) > 0.8
    assert result.get("error_message", "") == ""
