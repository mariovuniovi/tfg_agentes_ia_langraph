"""Message builders for the planner agent."""
from typing import Any
import json

from langchain_core.messages import HumanMessage


def format_planner_inputs(
    dataset_profile: dict[str, Any],
    task_metadata: dict[str, Any],
    problem_type: str,
) -> str:
    """Compact human-readable summary of inputs to seed the agent's reasoning."""
    return (
        f"problem_type: {problem_type}\n\n"
        f"task_metadata:\n{json.dumps(task_metadata, indent=2, default=str)}\n\n"
        f"dataset_profile:\n{json.dumps(dataset_profile, indent=2, default=str)}\n\n"
        f"Use the tools to retrieve evidence, then produce the PlannerOutput."
    )


def build_retry_message(last_error: str) -> HumanMessage:
    return HumanMessage(
        content=(
            f"Your previous PlannerOutput was rejected by validation: {last_error}\n\n"
            f"Produce a corrected PlannerOutput. You may call tools again as needed; "
            f"the retry uses a fresh tool trace."
        )
    )
