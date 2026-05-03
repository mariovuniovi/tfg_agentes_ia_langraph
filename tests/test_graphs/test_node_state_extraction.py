"""Tests for _extract_tool_json and updated worker node state extraction."""

import json
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

# ---------------------------------------------------------------------------
# _extract_tool_json
# ---------------------------------------------------------------------------


def test_extract_tool_json_finds_matching_message():
    from mlops_agents.graphs.mlops_graph import _extract_tool_json

    msgs = [
        ToolMessage(
            content='{"passed": true, "row_count": 150}',
            tool_call_id="call_1",
            name="check_data_quality",
        )
    ]
    result = _extract_tool_json(msgs, "check_data_quality")
    assert result == {"passed": True, "row_count": 150}


def test_extract_tool_json_returns_empty_dict_when_no_match():
    from mlops_agents.graphs.mlops_graph import _extract_tool_json

    result = _extract_tool_json([], "check_data_quality")
    assert result == {}


def test_extract_tool_json_returns_last_matching_message():
    from mlops_agents.graphs.mlops_graph import _extract_tool_json

    msgs = [
        ToolMessage(content='{"row_count": 100}', tool_call_id="1", name="check_data_quality"),
        ToolMessage(content='{"row_count": 200}', tool_call_id="2", name="check_data_quality"),
    ]
    result = _extract_tool_json(msgs, "check_data_quality")
    assert result["row_count"] == 200


def test_extract_tool_json_handles_invalid_json_gracefully():
    from mlops_agents.graphs.mlops_graph import _extract_tool_json

    msgs = [ToolMessage(content="not valid json", tool_call_id="1", name="my_tool")]
    result = _extract_tool_json(msgs, "my_tool")
    assert result == {}


def test_extract_tool_json_handles_list_result():
    from mlops_agents.graphs.mlops_graph import _extract_tool_json

    payload = json.dumps([{"run_id": "abc", "metrics": {"accuracy": 0.95}}])
    msgs = [ToolMessage(content=payload, tool_call_id="1", name="get_best_run")]
    result = _extract_tool_json(msgs, "get_best_run")
    assert isinstance(result, list)
    assert result[0]["run_id"] == "abc"


def test_extract_tool_json_skips_non_tool_messages():
    from mlops_agents.graphs.mlops_graph import _extract_tool_json

    msgs = [
        HumanMessage(content="run pipeline"),
        AIMessage(content="calling tool"),
        ToolMessage(content='{"found": true}', tool_call_id="1", name="my_tool"),
    ]
    result = _extract_tool_json(msgs, "my_tool")
    assert result == {"found": True}


# ---------------------------------------------------------------------------
# data_validator_node
# ---------------------------------------------------------------------------


def _make_state() -> dict:
    return {
        "messages": [HumanMessage(content="Run pipeline on iris.csv")],
        "next": "",
        "dataset_paths": ["./data/samples/iris.csv"],
        "dataset_path": "./data/samples/iris.csv",
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
        "agent_attempt_counts": {},
        "dataset_summary": {},
    }


def test_agent_state_has_dataset_summary_field():
    from mlops_agents.state.agent_state import AgentState
    import typing
    hints = typing.get_type_hints(AgentState)
    assert "dataset_summary" in hints


def test_data_validator_node_populates_validation_report():
    from mlops_agents.graphs.mlops_graph import data_validator_node

    quality_json = json.dumps(
        {
            "passed": True,
            "row_count": 150,
            "column_count": 5,
            "missing_values_total": 0,
            "max_missing_pct": 0.0,
            "duplicate_rows": 0,
        }
    )
    validation_json = json.dumps({"passed": True, "output_path": "./data/processed/iris.csv"})
    mock_result = {
        "messages": [
            ToolMessage(content=quality_json, tool_call_id="1", name="check_data_quality"),
            ToolMessage(content=validation_json, tool_call_id="2", name="validate_against_schema"),
            AIMessage(content="Data validation passed."),
        ]
    }
    with patch("mlops_agents.graphs.mlops_graph.get_agent") as mock_get_agent, \
         patch("mlops_agents.graphs.mlops_graph.interrupt", return_value={"approved": True, "comment": ""}):
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = mock_result
        mock_get_agent.return_value = mock_agent

        command = data_validator_node(_make_state())

    assert command.update["validation_report"]["passed"] is True
    assert command.update["validation_report"]["row_count"] == 150
    assert command.update["validation_passed"] is True
    assert command.goto == "supervisor"


def test_data_validator_node_passed_false_when_no_tool_output():
    from mlops_agents.graphs.mlops_graph import data_validator_node

    mock_result = {"messages": [AIMessage(content="Could not validate.")]}
    with patch("mlops_agents.graphs.mlops_graph.get_agent") as mock_get_agent:
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = mock_result
        mock_get_agent.return_value = mock_agent

        command = data_validator_node(_make_state())

    assert command.update["validation_report"] == {}
    assert command.update["validation_passed"] is False
    assert "error_message" in command.update
    assert len(command.update["error_message"]) > 0
    assert command.goto == "supervisor"


# ---------------------------------------------------------------------------
# trainer_node
# ---------------------------------------------------------------------------


def test_trainer_node_populates_training_metrics():
    from mlops_agents.graphs.mlops_graph import trainer_node

    train_json = json.dumps(
        {
            "model_type": "random_forest",
            "model_path": "./models/random_forest_model.pkl",
            "hyperparameters": {"n_estimators": 100},
            "train_accuracy": 0.98,
            "val_accuracy": 0.95,
            "classification_report": {},
        }
    )
    mlflow_json = json.dumps({"run_id": "abc123", "model_uri": "runs:/abc123/model"})
    mock_result = {
        "messages": [
            ToolMessage(content=train_json, tool_call_id="1", name="train_model"),
            ToolMessage(content=mlflow_json, tool_call_id="2", name="log_experiment"),
            AIMessage(content="Training complete."),
        ]
    }
    with patch("mlops_agents.graphs.mlops_graph.get_agent") as mock_get_agent:
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = mock_result
        mock_get_agent.return_value = mock_agent

        command = trainer_node(_make_state())

    assert command.update["training_metrics"]["model_type"] == "random_forest"
    assert command.update["training_metrics"]["val_accuracy"] == 0.95
    assert command.update["training_run_id"] == "abc123"
    assert command.update["trained_model_path"] == "./models/random_forest_model.pkl"
    assert command.goto == "supervisor"


# ---------------------------------------------------------------------------
# evaluator_node
# ---------------------------------------------------------------------------


def test_evaluator_node_populates_evaluation_report():
    from mlops_agents.graphs.mlops_graph import evaluator_node

    runs_json = json.dumps(
        [
            {
                "run_id": "run1",
                "metrics": {"accuracy": 0.97, "f1_score": 0.96},
                "params": {},
                "model_uri": "runs:/run1/model",
            },
            {
                "run_id": "run0",
                "metrics": {"accuracy": 0.93, "f1_score": 0.92},
                "params": {},
                "model_uri": "runs:/run0/model",
            },
        ]
    )
    mock_result = {
        "messages": [
            ToolMessage(content=runs_json, tool_call_id="1", name="get_best_run"),
            AIMessage(content="Candidate beats baseline. Recommend promote."),
        ]
    }
    with patch("mlops_agents.graphs.mlops_graph.get_agent") as mock_get_agent:
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = mock_result
        mock_get_agent.return_value = mock_agent

        command = evaluator_node(_make_state())

    assert command.update["evaluation_report"]["candidate_metrics"]["accuracy"] == 0.97
    assert command.update["evaluation_report"]["baseline_metrics"]["accuracy"] == 0.93
    assert command.update["evaluation_report"]["candidate_run_id"] == "run1"
    assert command.goto == "supervisor"


def test_data_validator_node_includes_imputation_in_hitl_payload():
    from mlops_agents.graphs.mlops_graph import data_validator_node

    imputation_json = json.dumps({
        "output_path": "./data/processed/iris.csv",
        "imputed_columns": {
            "sepal_width": {"strategy": "mean", "fill_value": 3.5, "rows_affected": 1}
        },
    })
    validation_json = json.dumps({"passed": True})
    mock_result = {
        "messages": [
            ToolMessage(content=validation_json, tool_call_id="1", name="validate_against_schema"),
            ToolMessage(content=imputation_json, tool_call_id="2", name="impute_missing_values"),
            AIMessage(content="Validation passed after imputation."),
        ]
    }

    captured_payload: dict = {}

    def fake_interrupt(payload: dict) -> dict:
        captured_payload.update(payload)
        return {"approved": True, "comment": ""}

    with patch("mlops_agents.graphs.mlops_graph.get_agent") as mock_get_agent, \
         patch("mlops_agents.graphs.mlops_graph.interrupt", side_effect=fake_interrupt):
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = mock_result
        mock_get_agent.return_value = mock_agent

        command = data_validator_node(_make_state())

    assert "imputation_applied" in captured_payload
    assert "sepal_width" in captured_payload["imputation_applied"]["imputed_columns"]
    assert command.update["validation_passed"] is True
    assert command.goto == "supervisor"


def test_data_validator_node_no_hitl_when_validation_fails():
    from mlops_agents.graphs.mlops_graph import data_validator_node

    validation_json = json.dumps({"passed": False, "violations": [{"column": "target", "rule": "allowed_values", "detail": "Unexpected values: ['bad']"}]})
    mock_result = {
        "messages": [
            ToolMessage(content=validation_json, tool_call_id="1", name="validate_against_schema"),
            AIMessage(content="Validation failed. Target column has invalid values."),
        ]
    }

    interrupt_called = []

    def fail_if_called(payload: dict) -> dict:
        interrupt_called.append(payload)
        return {}

    with patch("mlops_agents.graphs.mlops_graph.get_agent") as mock_get_agent, \
         patch("mlops_agents.graphs.mlops_graph.interrupt", side_effect=fail_if_called):
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = mock_result
        mock_get_agent.return_value = mock_agent

        command = data_validator_node(_make_state())

    assert len(interrupt_called) == 0
    assert command.update["validation_passed"] is False
    assert "error_message" in command.update
    assert command.goto == "supervisor"
