"""Pure helper functions for the Pipeline Streamlit page.

No Streamlit imports — these are extracted for testability.
"""

from langchain_core.messages import HumanMessage


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
        return None

    node_name = next(iter(event))
    return f"✅ `[{node_name}]` completed"


def build_initial_state(dataset_path: str) -> dict:
    """Build the initial LangGraph state dict for a pipeline run."""
    return {
        "messages": [HumanMessage(content=f"Run the full MLOps pipeline on dataset: {dataset_path}")],
        "next": "",
        "dataset_path": dataset_path,
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
