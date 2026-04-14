"""Unit tests for the supervisor routing node."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from mlops_agents.state.schemas import RouterOutput


def make_state(messages=None, **kwargs):
    base = {
        "messages": messages or [HumanMessage(content="Run the pipeline.")],
        "next": "",
        "dataset_path": "test.csv",
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
        "retry_count": 0,
    }
    base.update(kwargs)
    return base


@patch("mlops_agents.agents.supervisor._router_llm")
def test_supervisor_routes_to_data_validator_first(mock_llm):
    """Supervisor should route to data_validator at the start of a new pipeline."""
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = RouterOutput(
        next="data_validator",
        reasoning="Always start with data validation.",
    )
    mock_llm.with_structured_output.return_value = mock_structured

    from mlops_agents.agents.supervisor import supervisor_node

    state = make_state()
    command = supervisor_node(state)

    assert command.goto == "data_validator"
    assert command.update["next"] == "data_validator"


@patch("mlops_agents.agents.supervisor._router_llm")
def test_supervisor_routes_to_end_on_finish(mock_llm):
    """Supervisor should return END when LLM selects FINISH."""
    from langgraph.graph import END

    mock_structured = MagicMock()
    mock_structured.invoke.return_value = RouterOutput(
        next="FINISH",
        reasoning="Pipeline complete.",
    )
    mock_llm.with_structured_output.return_value = mock_structured

    from mlops_agents.agents.supervisor import supervisor_node

    state = make_state()
    command = supervisor_node(state)

    assert command.goto == END
