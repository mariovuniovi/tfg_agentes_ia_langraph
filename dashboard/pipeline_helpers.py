"""Pure helper functions for the Pipeline Streamlit page.

No Streamlit imports — these are extracted for testability.
"""

from pathlib import Path

import pandas as pd
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
        # Supervisor emitted a partial update without a routing decision — skip silently
        return None

    if not event:
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
