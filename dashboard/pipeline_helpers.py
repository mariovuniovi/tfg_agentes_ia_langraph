"""Pure helper functions for the Pipeline Streamlit page.

No Streamlit imports — these are extracted for testability.
"""

import time
from pathlib import Path
from typing import TypedDict

import pandas as pd
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage

_tool_start_times: dict[str, float] = {}


def reset_tool_start_times() -> None:
    """Clear the tool start times dict for a new pipeline run."""
    _tool_start_times.clear()


class PipelineEvent(TypedDict):
    type: str
    agent: str
    timestamp_ms: float
    data: dict


def event_to_log_line(event: dict) -> str | None:
    """Convert a LangGraph stream event dict to a UI log line.

    Returns None for events that should be silently skipped.

    Event shapes:
      {"supervisor": {"next": "data_validator", ...}}  → routing line
      {"supervisor": {"next": "FINISH", ...}}          → finish line
      {"data_validator": {...}}                         → worker line
      {"__interrupt__": [...]}                         → None (handled by caller)
    """
    if "__interrupt__" in event:
        return None

    if "supervisor" in event:
        next_agent = event["supervisor"].get("next", "")
        if next_agent == "FINISH":
            return "🏁 `[supervisor]` → **FINISH**"
        if next_agent:
            return f"🔀 `[supervisor]` → **{next_agent}**"
        # Supervisor emitted a partial update without a routing decision — skip silently
        return None

    if not event:
        return None
    node_name = next(iter(event))
    return f"✅ `[{node_name}]` completed"


def build_initial_state(dataset_paths: list[str]) -> dict:
    """Build the initial LangGraph state dict for a pipeline run."""
    paths_display = ", ".join(dataset_paths)
    return {
        "messages": [HumanMessage(content=f"Run the full MLOps pipeline on these raw files: {paths_display}")],
        "next": "",
        "dataset_paths": dataset_paths,
        "dataset_path": "",
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


def parse_stream_event(chunk: tuple) -> PipelineEvent | None:
    try:
        message_chunk, metadata = chunk
    except (TypeError, ValueError):
        return None

    agent: str = metadata.get("langgraph_node", "unknown") if isinstance(metadata, dict) else "unknown"
    now_ms: float = time.time() * 1000

    if isinstance(message_chunk, AIMessageChunk):
        tool_calls = message_chunk.tool_calls
        if tool_calls:
            tool_name: str = tool_calls[0]["name"]
            _tool_start_times[tool_name] = now_ms
            return PipelineEvent(
                type="tool_call",
                agent=agent,
                timestamp_ms=now_ms,
                data={"tool_name": tool_name, "arguments": tool_calls[0].get("args", {})},
            )
        if message_chunk.content:
            return PipelineEvent(
                type="agent_reasoning",
                agent=agent,
                timestamp_ms=now_ms,
                data={"content": message_chunk.content},
            )
        return None

    if isinstance(message_chunk, ToolMessage):
        tool_name = message_chunk.name or ""
        start_ms = _tool_start_times.pop(tool_name, now_ms)
        duration_ms: float = now_ms - start_ms
        return PipelineEvent(
            type="tool_result",
            agent=agent,
            timestamp_ms=now_ms,
            data={"tool_name": tool_name, "result": message_chunk.content, "duration_ms": duration_ms},
        )

    return None


def extract_panel_data(state: dict) -> dict:
    """Extract displayable panel data from a raw LangGraph state dict.

    Returns a dict with keys:
      validation_report  — dict from check_data_quality tool output
      training_metrics   — dict with model_type, train_accuracy, val_accuracy
      evaluation_report  — dict with candidate_metrics, baseline_metrics
      dataset_preview    — list of row dicts (first 10 rows), loaded once after validation
    All values are empty ({} / []) when the corresponding stage has not yet completed.
    """
    validation_report: dict = state.get("validation_report") or {}
    training_metrics: dict = state.get("training_metrics") or {}
    evaluation_report: dict = state.get("evaluation_report") or {}

    dataset_preview: list = []
    if validation_report:
        dataset_path = state.get("dataset_path", "")
        if dataset_path and Path(dataset_path).exists():
            try:
                dataset_preview = pd.read_csv(dataset_path).head(10).to_dict("records")
            except Exception:
                dataset_preview = []

    return {
        "validation_report": validation_report,
        "training_metrics": training_metrics,
        "evaluation_report": evaluation_report,
        "dataset_preview": dataset_preview,
    }
