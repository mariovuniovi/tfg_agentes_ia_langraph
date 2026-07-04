"""Context building and tool-result extraction for the data validation agent."""

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage

from mlops_agents.state.agent_state import AgentState


def extract_tool_json(messages: list, tool_name: str) -> Any:
    """Return the parsed JSON content of the last ToolMessage matching tool_name.

    Returns {} if no matching message is found or JSON parsing fails.
    Returns a list when the tool responded with a JSON array (e.g. get_best_run).
    """
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage) and getattr(msg, "name", None) == tool_name:
            try:
                return json.loads(msg.content)
            except (json.JSONDecodeError, TypeError):
                return {}
    return {}


def build_data_validator_context(
    state: AgentState,
    *,
    schema_json: str = "{}",
    schema_path: str = "",
) -> HumanMessage:
    from mlops_agents.tools.join_discovery_tools import profile_raw_datasets as _profile

    paths: list[str] = state.get("dataset_paths") or []

    # Build name → path mapping; use filename stem as dataset name
    raw_paths = {Path(p).stem: p for p in paths}

    profiles_section = ""
    multi_file_note = ""
    if len(paths) > 1:
        try:
            profiles = _profile(raw_paths)
            profiles_section = "\nRaw dataset profiles:\n" + json.dumps(
                [p.model_dump() for p in profiles], default=str, indent=2
            )
            multi_file_note = (
                "\n\nIMPORTANT: Multiple raw datasets are provided. "
                "You MUST use the join discovery workflow:\n"
                "  1. Propose join candidates based on the profiles above.\n"
                "  2. Call evaluate_join_candidates() to measure overlap and cardinality.\n"
                "  3. Call execute_join_plan() with your selections + the exact evaluations string.\n"
                "Do NOT call merge_datasets — that tool does not support inferred joins."
            )
        except Exception as exc:
            profiles_section = f"\n(Could not pre-profile raw files: {exc})"

    single_file_note = (
        "\nNOTE: Only ONE file was uploaded. "
        "Do NOT call merge_datasets or execute_join_plan. "
        "After load_dataset, go directly to apply_column_mapping on this single file."
        if len(paths) == 1 else ""
    )
    return HumanMessage(content=(
        f"Raw files: {json.dumps(paths)}\n"
        f"Schema path: {schema_path}\n"
        f"Target schema:\n{schema_json}"
        f"{profiles_section}"
        f"{multi_file_note}"
        f"{single_file_note}"
    ))
