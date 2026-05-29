"""Deterministic workflow controller — replaces the LLM supervisor.

Routes the pipeline by reading state fields. No LLM, no reasoning.
"""
from __future__ import annotations

from typing import Any

from langgraph.graph import END
from langgraph.types import Command

from mlops_agents.config.settings import settings
from mlops_agents.utils.logging import get_logger

logger = get_logger(__name__)


def workflow_controller(state: dict[str, Any]) -> Command:
    counts = state.get("agent_attempt_counts") or {}
    max_attempts = settings.max_attempts_per_agent

    if state.get("error_message"):
        return Command(goto=END)

    if not state.get("validation_passed"):
        if counts.get("data_validator", 0) >= max_attempts:
            return Command(
                goto=END,
                update={"error_message": "data_validator: max attempts reached"},
            )
        counts_next = {**counts, "data_validator": counts.get("data_validator", 0) + 1}
        return Command(
            goto="data_validator",
            update={"agent_attempt_counts": counts_next},
        )

    if state.get("dataset_approved") is None:
        return Command(goto="dataset_approval")

    if state.get("dataset_approved") is False:
        counts_next = {**counts, "data_validator": counts.get("data_validator", 0) + 1}
        if counts_next["data_validator"] > max_attempts:
            return Command(
                goto=END,
                update={"error_message": "Dataset rejected after max attempts"},
            )
        return Command(
            goto="data_validator",
            update={
                "dataset_approved": None,
                "validation_passed": False,
                "agent_attempt_counts": counts_next,
            },
        )

    if not state.get("training_plan"):
        return Command(goto="planner")

    if not state.get("training_run_id"):
        return Command(goto="executor")

    if state.get("evaluation_passed") is None:
        return Command(goto="evaluation")

    if state.get("evaluation_report_audit") is None:
        return Command(goto="report_writer")

    if state.get("evaluation_passed") is False:
        return Command(goto=END)            # deterministic rejection — audit written, no Gate 2

    if state.get("deployment_approved") is None:
        return Command(goto="deployment_approval")

    if state.get("deployment_approved") is False:
        return Command(goto=END)            # Gate 2 rejection — terminal

    if state.get("deployment_decision") == "pending":
        return Command(goto="deployer")

    return Command(goto=END)
