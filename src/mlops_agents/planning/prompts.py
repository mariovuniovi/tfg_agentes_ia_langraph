"""Message builders for the planner agent."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage


def format_planner_inputs(
    dataset_profile: dict[str, Any],
    task_metadata: dict[str, Any],
    problem_type: str,
    forecasting_policy_summary: str | None = None,
) -> str:
    """Compact human-readable summary of inputs to seed the agent's reasoning."""
    policy = ""
    if forecasting_policy_summary:
        policy = (
            "\n\nDeterministic forecasting policy (FIXED — do NOT choose validation or "
            f"exog strategy; select models suited to it):\n{forecasting_policy_summary}"
        )
    return (
        f"problem_type: {problem_type}\n\n"
        f"task_metadata:\n{json.dumps(task_metadata, indent=2, default=str)}\n\n"
        f"dataset_profile:\n{json.dumps(dataset_profile, indent=2, default=str)}"
        f"{policy}\n\n"
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
