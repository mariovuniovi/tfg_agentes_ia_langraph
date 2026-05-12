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
        "processed_dataset_path": "./data/samples/iris.csv",
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
        "problem_type": "",
        "task_metadata": {},
        "schema_json": json.dumps({
            "problem_type": "classification",
            "target_column": "target",
            "columns": [{"name": "target"}],
        }),
    }


def test_agent_state_has_dataset_summary_field():
    import typing

    from mlops_agents.state.agent_state import AgentState
    hints = typing.get_type_hints(AgentState)
    assert "dataset_summary" in hints


def test_agent_state_has_problem_type_field():
    import typing

    from mlops_agents.state.agent_state import AgentState
    hints = typing.get_type_hints(AgentState)
    assert "problem_type" in hints
    assert hints["problem_type"] is str


def test_agent_state_has_task_metadata_field():
    import typing

    from mlops_agents.state.agent_state import AgentState
    hints = typing.get_type_hints(AgentState)
    assert "task_metadata" in hints
    assert hints["task_metadata"] is dict


def test_agent_state_has_schema_json_field():
    import typing

    from mlops_agents.state.agent_state import AgentState
    hints = typing.get_type_hints(AgentState)
    assert "schema_json" in hints
    assert hints["schema_json"] is str


# ---------------------------------------------------------------------------
# _validate_schema_contract
# ---------------------------------------------------------------------------


def test_validate_schema_contract_passes_for_valid_classification():
    from mlops_agents.graphs.mlops_graph import _validate_schema_contract

    schema = {
        "problem_type": "classification",
        "target_column": "label",
        "columns": [
            {"name": "feature_a"},
            {"name": "label"},
        ],
    }
    _validate_schema_contract(schema)  # must not raise


def test_validate_schema_contract_passes_for_valid_regression():
    from mlops_agents.graphs.mlops_graph import _validate_schema_contract

    schema = {
        "problem_type": "regression",
        "target_column": "price",
        "columns": [{"name": "size"}, {"name": "price"}],
    }
    _validate_schema_contract(schema)  # must not raise


def test_validate_schema_contract_passes_for_valid_forecasting():
    from mlops_agents.graphs.mlops_graph import _validate_schema_contract

    schema = {
        "problem_type": "forecasting",
        "target_column": "sales",
        "datetime_column": "date",
        "series_id_columns": ["store_id"],
        "forecast_horizon": 30,
        "frequency": "D",
        "columns": [
            {"name": "date"},
            {"name": "store_id"},
            {"name": "sales"},
        ],
    }
    _validate_schema_contract(schema)  # must not raise


def test_validate_schema_contract_raises_on_missing_problem_type():
    import pytest

    from mlops_agents.graphs.mlops_graph import _validate_schema_contract

    schema = {"target_column": "label", "columns": [{"name": "label"}]}
    with pytest.raises(ValueError, match="problem_type"):
        _validate_schema_contract(schema)


def test_validate_schema_contract_raises_on_unknown_problem_type():
    import pytest

    from mlops_agents.graphs.mlops_graph import _validate_schema_contract

    schema = {
        "problem_type": "clustering",
        "target_column": "label",
        "columns": [{"name": "label"}],
    }
    with pytest.raises(ValueError, match="problem_type"):
        _validate_schema_contract(schema)


def test_validate_schema_contract_raises_when_target_column_missing_from_schema():
    import pytest

    from mlops_agents.graphs.mlops_graph import _validate_schema_contract

    schema = {
        "problem_type": "classification",
        "target_column": "nonexistent",
        "columns": [{"name": "feature_a"}],
    }
    with pytest.raises(ValueError, match="target_column"):
        _validate_schema_contract(schema)


def test_validate_schema_contract_raises_when_target_column_not_declared():
    import pytest

    from mlops_agents.graphs.mlops_graph import _validate_schema_contract

    schema = {
        "problem_type": "regression",
        "columns": [{"name": "price"}],
    }
    with pytest.raises(ValueError, match="target_column"):
        _validate_schema_contract(schema)


def test_validate_schema_contract_raises_on_missing_forecasting_fields():
    import pytest

    from mlops_agents.graphs.mlops_graph import _validate_schema_contract

    schema = {
        "problem_type": "forecasting",
        "target_column": "sales",
        "columns": [{"name": "sales"}],
        # missing datetime_column, forecast_horizon, frequency
    }
    with pytest.raises(ValueError, match="datetime_column|forecast_horizon|frequency"):
        _validate_schema_contract(schema)


def test_validate_schema_contract_raises_when_forecast_horizon_not_positive():
    import pytest

    from mlops_agents.graphs.mlops_graph import _validate_schema_contract

    schema = {
        "problem_type": "forecasting",
        "target_column": "sales",
        "datetime_column": "date",
        "forecast_horizon": 0,
        "frequency": "D",
        "columns": [{"name": "date"}, {"name": "sales"}],
    }
    with pytest.raises(ValueError, match="forecast_horizon"):
        _validate_schema_contract(schema)


def test_validate_schema_contract_raises_when_forecast_horizon_is_negative():
    import pytest

    from mlops_agents.graphs.mlops_graph import _validate_schema_contract

    schema = {
        "problem_type": "forecasting",
        "target_column": "sales",
        "datetime_column": "date",
        "forecast_horizon": -5,
        "frequency": "D",
        "columns": [{"name": "date"}, {"name": "sales"}],
    }
    with pytest.raises(ValueError, match="forecast_horizon"):
        _validate_schema_contract(schema)


def test_validate_schema_contract_raises_when_datetime_column_not_in_columns():
    import pytest

    from mlops_agents.graphs.mlops_graph import _validate_schema_contract

    schema = {
        "problem_type": "forecasting",
        "target_column": "sales",
        "datetime_column": "nonexistent_date",
        "forecast_horizon": 7,
        "frequency": "D",
        "columns": [{"name": "sales"}],
    }
    with pytest.raises(ValueError, match="datetime_column"):
        _validate_schema_contract(schema)


def test_validate_schema_contract_raises_when_series_id_column_not_in_columns():
    import pytest

    from mlops_agents.graphs.mlops_graph import _validate_schema_contract

    schema = {
        "problem_type": "forecasting",
        "target_column": "sales",
        "datetime_column": "date",
        "series_id_columns": ["missing_store"],
        "forecast_horizon": 7,
        "frequency": "D",
        "columns": [{"name": "date"}, {"name": "sales"}],
    }
    with pytest.raises(ValueError, match="series_id_columns"):
        _validate_schema_contract(schema)


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


def test_executor_node_returns_to_supervisor(tmp_path):
    """executor_node (deterministic) routes to supervisor and populates training state fields."""
    import pandas as pd
    from sklearn.datasets import load_iris
    from mlops_agents.graphs.mlops_graph import executor_node
    from mlops_agents.contracts.training import TrainingPlan, TrainingPlanCandidate, TrialBudget

    data = load_iris(as_frame=True)
    df = pd.concat([data.data, data.target.rename("target")], axis=1)
    csv_path = tmp_path / "iris.csv"
    df.to_csv(csv_path, index=False)

    plan = TrainingPlan(
        problem_type="classification",
        candidates=[TrainingPlanCandidate(priority=1, model_key="logistic_regression")],
        trial_budget=TrialBudget(total_trials=3, min_trials_per_candidate=3, max_trials_per_candidate=3),
    )
    state = _make_state()
    state["processed_dataset_path"] = str(csv_path)
    state["problem_type"] = "classification"
    state["task_metadata"] = {"target_column": "target", "problem_type": "classification"}
    state["training_plan"] = plan.model_dump()

    command = executor_node(state)

    assert command.goto == "supervisor"
    assert "trained_model_path" in command.update
    assert "training_run_id" in command.update
    assert "training_metrics" in command.update
    assert "champion_candidate" in command.update
    assert "experience_record_path" in command.update
    assert isinstance(command.update["training_metrics"], dict)


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


# ---------------------------------------------------------------------------
# Context builder pure functions
# ---------------------------------------------------------------------------


def test_build_data_validator_context_includes_raw_files():
    from mlops_agents.graphs.mlops_graph import _build_data_validator_context

    state = _make_state()
    msg = _build_data_validator_context(state)
    assert "./data/samples/iris.csv" in msg.content
    assert "Raw files:" in msg.content


def test_build_data_validator_context_includes_schema_path():
    from mlops_agents.graphs.mlops_graph import _build_data_validator_context

    state = _make_state()
    msg = _build_data_validator_context(
        state,
        schema_json='{"columns": [{"name": "target"}]}',
        schema_path="/data/schemas/iris.json",
    )
    assert "Schema path:" in msg.content
    assert "/data/schemas/iris.json" in msg.content
    assert "Target schema:" in msg.content
    assert '"target"' in msg.content


def test_build_evaluator_context_includes_run_id_and_metrics():
    from mlops_agents.graphs.mlops_graph import _build_evaluator_context

    state = _make_state()
    state["training_run_id"] = "abc123"
    state["trained_model_path"] = "models/rf.pkl"
    state["training_metrics"] = {"val_accuracy": 0.95}
    msg = _build_evaluator_context(state)
    assert "abc123" in msg.content
    assert "models/rf.pkl" in msg.content
    assert "0.95" in msg.content


def test_build_deployer_context_includes_model_uri_and_report():
    from mlops_agents.graphs.mlops_graph import _build_deployer_context

    state = _make_state()
    state["best_model_uri"] = "runs:/abc123/model"
    state["training_run_id"] = "abc123"
    state["evaluation_report"] = {"candidate_metrics": {"accuracy": 0.97}}
    msg = _build_deployer_context(state)
    assert "runs:/abc123/model" in msg.content
    assert "abc123" in msg.content
    assert "0.97" in msg.content


def test_data_validator_node_builds_dataset_summary_on_success():
    """data_validator_node must set dataset_summary in state when validation passes."""
    import os
    import tempfile

    from mlops_agents.graphs.mlops_graph import data_validator_node

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("a,b\n1,2\n3,4\n")
        tmp_path = f.name

    validation_json = json.dumps({"passed": True, "output_path": tmp_path})
    mock_result = {
        "messages": [
            ToolMessage(content=validation_json, tool_call_id="1", name="validate_against_schema"),
            AIMessage(content="Validation passed."),
        ]
    }

    try:
        with patch("mlops_agents.graphs.mlops_graph.get_agent") as mock_get_agent, \
             patch("mlops_agents.graphs.mlops_graph.interrupt", return_value={"approved": True, "comment": ""}):
            mock_agent = MagicMock()
            mock_agent.invoke.return_value = mock_result
            mock_get_agent.return_value = mock_agent

            state = _make_state()
            state["dataset_paths"] = [tmp_path]
            command = data_validator_node(state)
    finally:
        os.unlink(tmp_path)

    assert "dataset_summary" in command.update
    assert command.update["dataset_summary"]["row_count"] == 2
    assert "a" in command.update["dataset_summary"]["column_names"]
    assert "b" in command.update["dataset_summary"]["column_names"]


def test_data_validator_node_sets_empty_dataset_summary_on_failure():
    """dataset_summary must be {} when validation fails."""
    from mlops_agents.graphs.mlops_graph import data_validator_node

    mock_result = {"messages": [AIMessage(content="Could not validate.")]}
    with patch("mlops_agents.graphs.mlops_graph.get_agent") as mock_get_agent:
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = mock_result
        mock_get_agent.return_value = mock_agent

        command = data_validator_node(_make_state())

    assert command.update.get("dataset_summary") == {}


def test_data_validator_node_sets_problem_type_and_task_metadata_in_state():
    """data_validator_node must write problem_type and task_metadata to state after agent succeeds."""
    import os
    import tempfile

    from mlops_agents.graphs.mlops_graph import data_validator_node

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("sepal_length,target\n5.1,setosa\n6.3,versicolor\n")
        tmp_path = f.name

    schema = json.dumps({
        "problem_type": "classification",
        "target_column": "target",
        "columns": [{"name": "sepal_length"}, {"name": "target"}],
    })
    validation_json = json.dumps({"passed": True, "output_path": tmp_path})
    mock_result = {
        "messages": [
            ToolMessage(content=validation_json, tool_call_id="1", name="validate_against_schema"),
            AIMessage(content="Validation passed."),
        ]
    }

    try:
        with patch("mlops_agents.graphs.mlops_graph.get_agent") as mock_get_agent, \
             patch("mlops_agents.graphs.mlops_graph.interrupt", return_value={"approved": True, "comment": ""}):
            mock_agent = MagicMock()
            mock_agent.invoke.return_value = mock_result
            mock_get_agent.return_value = mock_agent

            state = _make_state()
            state["schema_json"] = schema
            command = data_validator_node(state)
    finally:
        os.unlink(tmp_path)

    assert command.update.get("problem_type") == "classification"
    assert command.update.get("task_metadata") == {"target_column": "target"}


def test_data_validator_node_aborts_on_contract_violation():
    """data_validator_node must return error Command immediately when schema contract is invalid."""
    from mlops_agents.graphs.mlops_graph import data_validator_node

    bad_schema = json.dumps({"columns": [{"name": "feature_a"}]})  # no problem_type

    interrupt_called = []

    def fail_if_called(payload: dict) -> dict:
        interrupt_called.append(payload)
        return {}

    with patch("mlops_agents.graphs.mlops_graph.get_agent") as mock_get_agent, \
         patch("mlops_agents.graphs.mlops_graph.interrupt", side_effect=fail_if_called):
        mock_agent = MagicMock()
        mock_get_agent.return_value = mock_agent

        state = _make_state()
        state["schema_json"] = bad_schema
        command = data_validator_node(state)

    mock_agent.invoke.assert_not_called()
    assert len(interrupt_called) == 0
    assert "problem_type" in command.update.get("error_message", "")
    assert command.update.get("validation_passed") is False
    assert command.update.get("problem_type") == ""
    assert command.update.get("task_metadata") == {}
    assert command.goto == "supervisor"


def test_data_validator_node_invokes_agent_with_isolated_context():
    """data_validator_node must NOT pass state['messages'] to agent.invoke."""
    from mlops_agents.graphs.mlops_graph import data_validator_node

    validation_json = json.dumps({"passed": True, "output_path": ""})
    mock_result = {
        "messages": [
            ToolMessage(content=validation_json, tool_call_id="1", name="validate_against_schema"),
            AIMessage(content="Validation passed."),
        ]
    }
    with patch("mlops_agents.graphs.mlops_graph.get_agent") as mock_get_agent, \
         patch("mlops_agents.graphs.mlops_graph.interrupt", return_value={"approved": True, "comment": ""}):
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = mock_result
        mock_get_agent.return_value = mock_agent

        state = _make_state()
        state["messages"] = [
            HumanMessage(content="Prior supervisor message 1"),
            HumanMessage(content="Prior supervisor message 2"),
        ]
        data_validator_node(state)

    call_messages = mock_agent.invoke.call_args[0][0]["messages"]
    assert len(call_messages) == 1, (
        f"Expected exactly 1 context message, got {len(call_messages)}. "
        "Prior state['messages'] must not be forwarded to worker agents."
    )


def test_executor_node_uses_plan_from_state(tmp_path):
    """executor_node must use training_plan from state (planner-generated)."""
    import pandas as pd
    from sklearn.datasets import load_iris
    from mlops_agents.graphs.mlops_graph import executor_node
    from mlops_agents.contracts.training import TrainingPlan, TrainingPlanCandidate, TrialBudget

    data = load_iris(as_frame=True)
    df = pd.concat([data.data, data.target.rename("target")], axis=1)
    csv_path = tmp_path / "iris.csv"
    df.to_csv(csv_path, index=False)

    plan = TrainingPlan(
        problem_type="classification",
        candidates=[TrainingPlanCandidate(priority=1, model_key="logistic_regression")],
        trial_budget=TrialBudget(total_trials=3, min_trials_per_candidate=3, max_trials_per_candidate=3),
    )
    state = _make_state()
    state["processed_dataset_path"] = str(csv_path)
    state["problem_type"] = "classification"
    state["task_metadata"] = {"target_column": "target", "problem_type": "classification"}
    state["training_plan"] = plan.model_dump()

    command = executor_node(state)

    assert command.goto == "supervisor"
    assert command.update["training_plan"]["candidates"][0]["model_key"] == "logistic_regression"


def test_evaluator_node_invokes_agent_with_isolated_context():
    """evaluator_node must pass exactly one context message — not state['messages']."""
    from mlops_agents.graphs.mlops_graph import evaluator_node

    runs_json = json.dumps([
        {"run_id": "run1", "metrics": {"accuracy": 0.97}, "params": {}, "model_uri": "runs:/run1/model"},
    ])
    mock_result = {
        "messages": [
            ToolMessage(content=runs_json, tool_call_id="1", name="get_best_run"),
            AIMessage(content="Evaluation complete."),
        ]
    }
    with patch("mlops_agents.graphs.mlops_graph.get_agent") as mock_get_agent:
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = mock_result
        mock_get_agent.return_value = mock_agent

        state = _make_state()
        state["messages"] = [HumanMessage(content="prior msg 1"), HumanMessage(content="prior msg 2")]
        evaluator_node(state)

    call_messages = mock_agent.invoke.call_args[0][0]["messages"]
    assert len(call_messages) == 1
    assert "Training run ID:" in call_messages[0].content


def test_deployer_node_invokes_agent_with_isolated_context():
    """deployer_node must pass exactly one context message — not state['messages']."""
    from mlops_agents.graphs.mlops_graph import deployer_node

    mock_result = {
        "messages": [AIMessage(content="Model registered as challenger.")]
    }
    with patch("mlops_agents.graphs.mlops_graph.get_agent") as mock_get_agent, \
         patch("mlops_agents.graphs.mlops_graph.interrupt", return_value={"approved": True}):
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = mock_result
        mock_get_agent.return_value = mock_agent

        state = _make_state()
        state["messages"] = [HumanMessage(content="prior 1"), HumanMessage(content="prior 2")]
        deployer_node(state)

    call_messages = mock_agent.invoke.call_args[0][0]["messages"]
    assert len(call_messages) == 1
    assert "Best model URI:" in call_messages[0].content


def test_build_evaluator_context_includes_problem_type_and_task_metadata():
    from mlops_agents.graphs.mlops_graph import _build_evaluator_context

    state = _make_state()
    state["problem_type"] = "regression"
    state["task_metadata"] = {"target_column": "price"}
    msg = _build_evaluator_context(state)
    assert "Problem type: regression" in msg.content
    assert "target_column" in msg.content


def test_build_deployer_context_includes_problem_type():
    from mlops_agents.graphs.mlops_graph import _build_deployer_context

    state = _make_state()
    state["problem_type"] = "forecasting"
    msg = _build_deployer_context(state)
    assert "Problem type: forecasting" in msg.content


def test_data_validator_node_reads_schema_json_from_state():
    """data_validator_node must use state['schema_json'] instead of loading from disk."""
    import os
    import tempfile

    from mlops_agents.graphs.mlops_graph import data_validator_node

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("sepal_length,target\n5.1,setosa\n6.3,versicolor\n")
        tmp_path = f.name

    schema = json.dumps({
        "problem_type": "classification",
        "target_column": "target",
        "columns": [{"name": "sepal_length", "dtype": "float"}, {"name": "target", "dtype": "str"}],
    })
    validation_json = json.dumps({"passed": True, "output_path": tmp_path})
    mock_result = {
        "messages": [
            ToolMessage(content=validation_json, tool_call_id="1", name="validate_against_schema"),
            AIMessage(content="Validation passed."),
        ]
    }

    try:
        with patch("mlops_agents.graphs.mlops_graph.get_agent") as mock_get_agent, \
             patch("mlops_agents.graphs.mlops_graph.interrupt", return_value={"approved": True, "comment": ""}):
            mock_agent = MagicMock()
            mock_agent.invoke.return_value = mock_result
            mock_get_agent.return_value = mock_agent

            state = _make_state()
            state["schema_json"] = schema
            command = data_validator_node(state)
    finally:
        os.unlink(tmp_path)

    assert command.update.get("problem_type") == "classification"
    assert command.update.get("task_metadata") == {"target_column": "target"}


def test_data_validator_node_aborts_when_schema_json_empty():
    """data_validator_node must abort immediately when schema_json is empty."""
    from mlops_agents.graphs.mlops_graph import data_validator_node

    with patch("mlops_agents.graphs.mlops_graph.get_agent") as mock_get_agent, \
         patch("mlops_agents.graphs.mlops_graph.interrupt") as mock_interrupt:
        mock_agent = MagicMock()
        mock_get_agent.return_value = mock_agent

        state = _make_state()
        state["schema_json"] = ""  # no schema uploaded
        command = data_validator_node(state)

    mock_agent.invoke.assert_not_called()
    mock_interrupt.assert_not_called()
    assert command.update.get("validation_passed") is False
    assert "schema" in command.update.get("error_message", "").lower()
    assert command.goto == "supervisor"
