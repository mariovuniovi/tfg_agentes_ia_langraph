"""Unit tests for Pipeline page helper functions."""

import pandas as pd
from langchain_core.messages import HumanMessage

from dashboard.pipeline_helpers import build_initial_state, event_to_log_line, extract_panel_data


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


def test_build_initial_state_sets_dataset_paths():
    paths = ["./data/samples/iris_measurements.csv", "./data/samples/iris_labels.csv"]
    state = build_initial_state(paths)
    assert state["dataset_paths"] == paths
    assert state["dataset_path"] == ""


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
    assert state["evaluation_passed"] is False


def test_event_to_log_line_empty_event_returns_none():
    assert event_to_log_line({}) is None


def test_event_to_log_line_worker_node_deployer():
    event = {"deployer": {"messages": []}}
    assert event_to_log_line(event) == "✅ `[deployer]` completed"


def test_extract_panel_data_empty_state_returns_all_empty():
    result = extract_panel_data({})
    assert result["validation_report"] == {}
    assert result["training_metrics"] == {}
    assert result["evaluation_report"] == {}
    assert result["dataset_preview"] == []


def test_extract_panel_data_returns_validation_report():
    report = {"passed": True, "row_count": 150, "column_count": 5}
    result = extract_panel_data({"validation_report": report, "dataset_path": ""})
    assert result["validation_report"] == report


def test_extract_panel_data_returns_training_metrics():
    metrics = {"model_type": "random_forest", "val_accuracy": 0.95}
    result = extract_panel_data({"training_metrics": metrics})
    assert result["training_metrics"] == metrics


def test_extract_panel_data_returns_evaluation_report():
    report = {"candidate_metrics": {"accuracy": 0.97}, "baseline_metrics": {}}
    result = extract_panel_data({"evaluation_report": report})
    assert result["evaluation_report"] == report


def test_extract_panel_data_no_preview_when_validation_report_empty():
    result = extract_panel_data({"validation_report": {}, "dataset_path": "./data/samples/iris.csv"})
    assert result["dataset_preview"] == []


def test_extract_panel_data_no_preview_when_path_missing():
    result = extract_panel_data({"validation_report": {"passed": True}, "dataset_path": "/nonexistent/file.csv"})
    assert result["dataset_preview"] == []


def test_extract_panel_data_no_preview_when_path_empty():
    result = extract_panel_data({"validation_report": {"passed": True}, "dataset_path": ""})
    assert result["dataset_preview"] == []


def test_extract_panel_data_loads_preview_when_valid(tmp_path):
    csv_path = tmp_path / "sample.csv"
    df = pd.DataFrame({"a": range(20), "b": range(20), "target": range(20)})
    df.to_csv(csv_path, index=False)
    result = extract_panel_data(
        {
            "validation_report": {"passed": True},
            "dataset_path": str(csv_path),
        }
    )
    assert len(result["dataset_preview"]) == 10
    assert result["dataset_preview"][0]["a"] == 0
