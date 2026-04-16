"""Unit tests for Pipeline page helper functions."""

from langchain_core.messages import HumanMessage

from dashboard.pipeline_helpers import build_initial_state, event_to_log_line


def test_event_to_log_line_supervisor_routes_to_agent():
    event = {"supervisor": {"next": "data_validator", "messages": []}}
    assert event_to_log_line(event) == "🔀 `[supervisor]` → **data_validator**"


def test_event_to_log_line_supervisor_finish():
    event = {"supervisor": {"next": "FINISH", "messages": []}}
    assert event_to_log_line(event) == "🏁 `[supervisor]` → **FINISH**"


def test_event_to_log_line_worker_node():
    event = {"data_validator": {"messages": []}}
    assert event_to_log_line(event) == "✅ `[data_validator]` completed"


def test_event_to_log_line_interrupt_returns_none():
    assert event_to_log_line({"__interrupt__": []}) is None


def test_event_to_log_line_supervisor_no_next_returns_none():
    assert event_to_log_line({"supervisor": {"messages": []}}) is None


def test_build_initial_state_sets_dataset_path():
    state = build_initial_state("./data/samples/iris.csv")
    assert state["dataset_path"] == "./data/samples/iris.csv"


def test_build_initial_state_has_human_message():
    state = build_initial_state("./data/samples/iris.csv")
    assert len(state["messages"]) == 1
    assert isinstance(state["messages"][0], HumanMessage)
    assert "iris.csv" in state["messages"][0].content


def test_build_initial_state_deployment_pending():
    state = build_initial_state("./data/samples/iris.csv")
    assert state["deployment_decision"] == "pending"


def test_build_initial_state_validation_false():
    state = build_initial_state("./data/samples/iris.csv")
    assert state["validation_passed"] is False
    assert state["evaluation_passed"] is False


def test_event_to_log_line_empty_event_returns_none():
    assert event_to_log_line({}) is None


def test_event_to_log_line_worker_node_deployer():
    event = {"deployer": {"messages": []}}
    assert event_to_log_line(event) == "✅ `[deployer]` completed"
