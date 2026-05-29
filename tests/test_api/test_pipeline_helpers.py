"""Unit tests for api.services.pipeline_helpers."""

from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage

import api.services.pipeline_helpers as _ph
from api.services.pipeline_helpers import build_initial_state, parse_stream_event


def test_build_initial_state_sets_dataset_paths():
    paths = ["./data/samples/iris_measurements.csv", "./data/samples/iris_labels.csv"]
    state = build_initial_state(paths)
    assert state["dataset_paths"] == paths
    assert state["processed_dataset_path"] == ""


def test_build_initial_state_has_human_message():
    paths = ["./data/samples/iris_measurements.csv", "./data/samples/iris_labels.csv"]
    state = build_initial_state(paths)
    assert len(state["messages"]) == 1
    assert isinstance(state["messages"][0], HumanMessage)
    assert "iris_measurements.csv" in state["messages"][0].content


def test_build_initial_state_deployment_pending():
    state = build_initial_state(["./data/samples/iris.csv"])
    assert state["deployment_decision"] == "pending"


def test_build_initial_state_validation_false():
    state = build_initial_state(["./data/samples/iris.csv"])
    assert state["validation_passed"] is False
    assert state["evaluation_passed"] is None


def test_build_initial_state_includes_schema_json():
    schema = '{"problem_type": "classification"}'
    state = build_initial_state(["./data/samples/iris.csv"], schema_json=schema)
    assert state["schema_json"] == schema


def test_build_initial_state_schema_json_defaults_to_empty():
    state = build_initial_state(["./data/samples/iris.csv"])
    assert state["schema_json"] == ""


def test_build_initial_state_includes_problem_type_and_task_metadata():
    state = build_initial_state(["./data/samples/iris.csv"])
    assert state["problem_type"] == ""
    assert state["task_metadata"] == {}


def test_build_initial_state_includes_dataset_summary():
    state = build_initial_state(["./data/samples/iris.csv"])
    assert state["dataset_summary"] == {}


def test_parse_stream_event_agent_reasoning():
    chunk = AIMessageChunk(content="thinking about data...")
    metadata = {"langgraph_node": "data_validator"}
    result = parse_stream_event((chunk, metadata))
    assert result is not None
    assert result["type"] == "agent_reasoning"
    assert result["agent"] == "data_validator"
    assert result["data"]["content"] == "thinking about data..."


def test_parse_stream_event_tool_call():
    chunk = AIMessageChunk(
        content="",
        tool_calls=[{"name": "load_dataset", "args": {"path": "iris.csv"}, "id": "call_1"}],
    )
    metadata = {"langgraph_node": "data_validator"}
    result = parse_stream_event((chunk, metadata))
    assert result is not None
    assert result["type"] == "tool_call"
    assert result["data"]["tool_name"] == "load_dataset"
    assert result["data"]["arguments"] == {"path": "iris.csv"}


def test_parse_stream_event_tool_result_with_duration():
    call_chunk = AIMessageChunk(
        content="",
        tool_calls=[{"name": "load_dataset", "args": {"path": "iris.csv"}, "id": "call_1"}],
    )
    _ph._tool_start_times.clear()
    parse_stream_event((call_chunk, {"langgraph_node": "data_validator"}))
    assert "load_dataset" in _ph._tool_start_times

    tool_msg = ToolMessage(content='{"rows": 150}', tool_call_id="call_1", name="load_dataset")
    result = parse_stream_event((tool_msg, {"langgraph_node": "data_validator"}))
    assert result is not None
    assert result["type"] == "tool_result"
    assert result["data"]["tool_name"] == "load_dataset"
    assert isinstance(result["data"]["duration_ms"], float)
    assert "load_dataset" not in _ph._tool_start_times


def test_parse_stream_event_unknown_chunk_returns_none():
    result = parse_stream_event(("not_a_message", {"langgraph_node": "supervisor"}))
    assert result is None


def test_parse_stream_event_empty_content_no_tool_calls_returns_none():
    chunk = AIMessageChunk(content="")
    result = parse_stream_event((chunk, {"langgraph_node": "data_validator"}))
    assert result is None


def test_build_initial_state_has_all_required_controller_fields():
    from api.services.pipeline_helpers import build_initial_state
    state = build_initial_state(["a.csv"])
    assert state["dataset_approved"] is None
    assert state["deployment_approved"] is None
    assert state["evaluation_passed"] is None
    assert state["evaluation_report_audit"] is None
    assert state["dataset_rejection_comment"] == ""
    assert state["evaluation_report_audit_status"] == ""
    assert state["candidate_metrics"] == {}
    assert state["champion_metrics"] == {}
    assert state["thresholds_applied"] == {}
