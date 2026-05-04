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
    """Graph should contain all 5 expected nodes."""
    from mlops_agents.graphs.mlops_graph import graph

    node_names = set(graph.nodes.keys())
    assert "supervisor" in node_names
    assert "data_validator" in node_names
    assert "trainer" in node_names
    assert "evaluator" in node_names
    assert "deployer" in node_names


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
        "dataset_path": processed,
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
        tool_content=f'{{"output_path": "{state["dataset_path"]}", "mapped_columns": 2}}',
    )

    with patch("mlops_agents.graphs.mlops_graph.get_agent", return_value=mock_agent), \
         patch("mlops_agents.graphs.mlops_graph.interrupt", return_value={"approved": True, "comment": ""}):
        from mlops_agents.graphs.mlops_graph import data_validator_node
        command = data_validator_node(state)

    assert command.goto == "supervisor"
    assert command.update["validation_passed"] is False  # tool returned empty {}


def test_data_validator_node_rejected_injects_message(tmp_path):
    """On rejection, node injects a HumanMessage with the comment and sets validation_passed=False."""
    state = _make_validator_state(tmp_path)
    mock_agent = _make_mock_agent(
        tool_name="validate_against_schema",
        tool_content='{"passed": true}',
    )

    with patch("mlops_agents.graphs.mlops_graph.get_agent", return_value=mock_agent), \
         patch("mlops_agents.graphs.mlops_graph.interrupt", return_value={"approved": False, "comment": "rename column X"}):
        from mlops_agents.graphs.mlops_graph import data_validator_node
        command = data_validator_node(state)

    assert command.goto == "supervisor"
    assert command.update["validation_passed"] is False
    rejection_msgs = [
        m for m in command.update["messages"]
        if isinstance(m, HumanMessage) and "rename column X" in m.content
    ]
    assert len(rejection_msgs) == 1


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
