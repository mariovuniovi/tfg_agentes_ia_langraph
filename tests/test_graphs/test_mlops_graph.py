"""Tests for the main MLOps graph structure."""

import json
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, ToolMessage


def test_graph_compiles_without_error():
    """The main graph should compile successfully on import."""
    from mlops_agents.graphs.mlops_graph import graph

    assert graph is not None


def test_graph_has_expected_nodes():
    """Graph should contain all expected nodes including planner and executor (SP5)."""
    from mlops_agents.graphs.mlops_graph import graph

    node_names = set(graph.nodes.keys())
    assert "supervisor" in node_names
    assert "data_validator" in node_names
    assert "planner" in node_names
    assert "executor" in node_names
    assert "evaluator" in node_names
    assert "deployer" in node_names
    assert "trainer" not in node_names


def _make_validator_state(tmp_path=None):
    from pathlib import Path
    import pandas as pd

    processed = ""
    if tmp_path:
        p = Path(tmp_path) / "processed.csv"
        pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(p, index=False)
        processed = str(p)

    return {
        "messages": [HumanMessage(content="Run pipeline.")],
        "next": "",
        "dataset_paths": ["data/samples/iris_measurements.csv"],
        "processed_dataset_path": processed,
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
        "schema_json": json.dumps({
            "problem_type": "classification",
            "target_column": "target",
            "columns": [{"name": "target"}],
        }),
    }


def _make_mock_agent(final_content="done", tool_name=None, tool_content="{}"):
    mock_agent = MagicMock()
    messages = [HumanMessage(content="context")]
    if tool_name:
        tm = ToolMessage(content=tool_content, tool_call_id="t1", name=tool_name)
        messages.append(tm)
    messages.append(HumanMessage(content=final_content))
    mock_agent.invoke.return_value = {"messages": messages}
    return mock_agent


def test_data_validator_node_approved_returns_to_supervisor(tmp_path):
    """On approval, node returns Command(goto='supervisor') with validation fields."""
    state = _make_validator_state(tmp_path)
    mock_agent = _make_mock_agent(
        tool_name="apply_column_mapping",
        tool_content=f'{{"output_path": "{state["processed_dataset_path"]}", "mapped_columns": 2}}',
    )

    with patch("mlops_agents.graphs.mlops_graph.get_agent", return_value=mock_agent), \
         patch("mlops_agents.graphs.mlops_graph.interrupt", return_value={"approved": True, "comment": ""}):
        from mlops_agents.graphs.mlops_graph import data_validator_node
        command = data_validator_node(state)

    assert command.goto == "supervisor"
    assert command.update["validation_passed"] is False  # tool returned empty {}


def test_data_validator_node_rejected_routes_to_data_validator(tmp_path):
    """First rejection (attempt < max) routes back to data_validator for retry, not supervisor."""
    state = _make_validator_state(tmp_path)
    mock_agent = _make_mock_agent(
        tool_name="validate_against_schema",
        tool_content='{"passed": true}',
    )

    with patch("mlops_agents.graphs.mlops_graph.get_agent", return_value=mock_agent), \
         patch("mlops_agents.graphs.mlops_graph.interrupt", return_value={"approved": False, "comment": "rename column X"}):
        from mlops_agents.graphs.mlops_graph import data_validator_node
        command = data_validator_node(state)

    assert command.goto == "data_validator"
    assert command.update["validation_passed"] is False
    assert command.update["error_message"] == ""
    rejection_msgs = [
        m for m in command.update["messages"]
        if isinstance(m, HumanMessage) and "rename column X" in m.content
    ]
    assert len(rejection_msgs) == 1


def test_data_validator_node_rejected_at_max_attempts_routes_to_supervisor(tmp_path):
    """Rejection on the final attempt routes to supervisor with error_message set."""
    state = _make_validator_state(tmp_path)
    state["agent_attempt_counts"] = {"data_validator": 3}  # attempt == max
    mock_agent = _make_mock_agent(
        tool_name="validate_against_schema",
        tool_content='{"passed": true}',
    )

    with patch("mlops_agents.graphs.mlops_graph.get_agent", return_value=mock_agent), \
         patch("mlops_agents.graphs.mlops_graph.interrupt", return_value={"approved": False, "comment": "still bad"}):
        from mlops_agents.graphs.mlops_graph import data_validator_node
        command = data_validator_node(state)

    assert command.goto == "supervisor"
    assert command.update["validation_passed"] is False
    assert "rejected after" in command.update["error_message"]
    assert "still bad" in command.update["error_message"]


def test_data_validator_node_retry_increments_attempt_count(tmp_path):
    """Retry Command must increment agent_attempt_counts['data_validator']."""
    state = _make_validator_state(tmp_path)
    state["agent_attempt_counts"] = {"data_validator": 1}
    mock_agent = _make_mock_agent(
        tool_name="validate_against_schema",
        tool_content='{"passed": true}',
    )

    with patch("mlops_agents.graphs.mlops_graph.get_agent", return_value=mock_agent), \
         patch("mlops_agents.graphs.mlops_graph.interrupt", return_value={"approved": False, "comment": ""}):
        from mlops_agents.graphs.mlops_graph import data_validator_node
        command = data_validator_node(state)

    assert command.update["agent_attempt_counts"]["data_validator"] == 2


def test_data_validator_node_retry_injects_prior_feedback(tmp_path):
    """On a retry attempt the agent invocation must include prior rejection feedback."""
    state = _make_validator_state(tmp_path)
    state["agent_attempt_counts"] = {"data_validator": 2}
    state["messages"] = [
        HumanMessage(content="Dataset rejected by human reviewer. Comment: drop nulls", name="data_validator"),
    ]
    captured_messages = []
    mock_agent = _make_mock_agent(
        tool_name="validate_against_schema",
        tool_content='{"passed": true}',
    )

    original_invoke = mock_agent.invoke

    def capturing_invoke(payload):
        captured_messages.extend(payload["messages"])
        return original_invoke(payload)

    mock_agent.invoke = capturing_invoke

    with patch("mlops_agents.graphs.mlops_graph.get_agent", return_value=mock_agent), \
         patch("mlops_agents.graphs.mlops_graph.interrupt", return_value={"approved": True, "comment": ""}):
        from mlops_agents.graphs.mlops_graph import data_validator_node
        data_validator_node(state)

    feedback_msgs = [m for m in captured_messages if "rejected" in m.content.lower()]
    assert len(feedback_msgs) >= 1, "Agent should receive prior rejection feedback on retry"


def test_data_validator_node_interrupt_payload_has_type(tmp_path):
    """The interrupt payload must have type='data_validation'."""
    state = _make_validator_state(tmp_path)
    mock_agent = _make_mock_agent(
        tool_name="validate_against_schema",
        tool_content='{"passed": true}',
    )
    captured = {}

    def fake_interrupt(value):
        captured["payload"] = value
        return {"approved": True, "comment": ""}

    with patch("mlops_agents.graphs.mlops_graph.get_agent", return_value=mock_agent), \
         patch("mlops_agents.graphs.mlops_graph.interrupt", side_effect=fake_interrupt):
        from mlops_agents.graphs.mlops_graph import data_validator_node
        data_validator_node(state)

    assert captured["payload"]["type"] == "data_validation"
    assert "dataset_preview" in captured["payload"]
    assert "validation_summary" in captured["payload"]
    assert "attempt" in captured["payload"]


# ---------------------------------------------------------------------------
# executor_node tests
# ---------------------------------------------------------------------------

def _make_executor_state(tmp_path, problem_type="classification"):
    """Minimal state for executor_node: valid CSV + training plan, no problem_type in task_metadata."""
    import pandas as pd
    from sklearn.datasets import load_iris

    data = load_iris(as_frame=True)
    df = pd.concat([data.data, data.target.rename("target")], axis=1)
    csv = tmp_path / "iris.csv"
    df.to_csv(csv, index=False)

    plan = {
        "problem_type": problem_type,
        "metric_to_optimize": None,
        "candidates": [
            {"priority": 1, "model_key": "logistic_regression", "initial_hyperparameters": {},
             "search_space_override": None, "requested_trials": None, "reason": ""},
        ],
        "models_not_recommended": [],
        "trial_budget": {"total_trials": 10, "allocation_strategy": "priority_weighted",
                         "max_trials_per_candidate": 10, "min_trials_per_candidate": 3},
        "forecasting_settings": None,
    }
    return {
        "messages": [],
        "processed_dataset_path": str(csv),
        "problem_type": problem_type,
        "task_metadata": {"target_column": "target"},  # no problem_type intentionally
        "training_plan": plan,
        "training_run_id": "",
        "trained_model_path": "",
        "training_metrics": {},
        "error_message": "",
    }


def test_executor_node_raises_without_training_plan(tmp_path):
    """executor_node must raise RuntimeError when training_plan is None."""
    import pandas as pd
    from mlops_agents.graphs.mlops_graph import executor_node

    csv = tmp_path / "data.csv"
    pd.DataFrame({"f": [1, 2], "target": [0, 1]}).to_csv(csv, index=False)
    state = {
        "messages": [],
        "processed_dataset_path": str(csv),
        "problem_type": "classification",
        "task_metadata": {"target_column": "target"},
        "training_plan": None,
        "error_message": "",
    }
    with pytest.raises(RuntimeError, match="training_plan"):
        executor_node(state)


def test_executor_node_injects_problem_type_into_task_metadata(tmp_path, monkeypatch):
    """executor_node must add problem_type from state into task_metadata before calling run_training_plan."""
    from unittest.mock import patch, MagicMock
    from mlops_agents.contracts.training import TrainingResult

    state = _make_executor_state(tmp_path)
    captured = {}

    mock_result = MagicMock(spec=TrainingResult)
    mock_result.champion_model_path = str(tmp_path / "model.pkl")
    mock_result.train_pool_path = str(tmp_path / "train.csv")
    mock_result.test_path = str(tmp_path / "test.csv")
    mock_result.split_metadata_path = str(tmp_path / "meta.json")
    mock_result.mlflow_parent_run_id = "run123"
    mock_result.champion_metrics = {"macro_f1": 0.9}
    mock_result.champion_candidate = {"model_key": "logistic_regression"}
    mock_result.experience_record_path = str(tmp_path / "exp.json")

    def capturing_run(plan, processed_dataset_path, target_column, task_metadata, **kwargs):
        captured["task_metadata"] = task_metadata
        return mock_result

    with patch("mlops_agents.training.executor.run_training_plan", side_effect=capturing_run):
        from mlops_agents.graphs.mlops_graph import executor_node
        executor_node(state)

    assert captured["task_metadata"]["problem_type"] == "classification", (
        "executor_node must merge problem_type from state into task_metadata"
    )
