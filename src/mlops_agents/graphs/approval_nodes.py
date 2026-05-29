"""HITL approval nodes — pure interrupt() wrappers.

Each node pauses the graph with interrupt() and writes an approval flag
to state. Payload `type` values are preserved from the old embedded HITL
so the existing SSE event shape stays backward compatible.
"""
from __future__ import annotations

from typing import Any

from langgraph.types import Command, interrupt

from mlops_agents.utils.logging import get_logger

logger = get_logger(__name__)


def dataset_approval_node(state: dict[str, Any]) -> Command:
    counts = state.get("agent_attempt_counts") or {}
    attempt = counts.get("data_validator", 1)
    approval = interrupt({
        "type": "data_validation",
        "question": "Review the processed dataset before training begins.",
        "attempt": attempt,
        "preview": state.get("dataset_summary", {}),
        "validation_report": state.get("validation_report", {}),
    })
    approved = bool(approval.get("approved", False))
    comment = approval.get("comment", "")
    logger.info(f"[gate1] dataset_approved={approved} comment={comment!r}")
    return Command(
        goto="workflow_controller",
        update={
            "dataset_approved": approved,
            "dataset_rejection_comment": "" if approved else comment,
        },
    )


def deployment_approval_node(state: dict[str, Any]) -> Command:
    approval = interrupt({
        "type": "deployer",
        "question": "Approve deployment of this model based on the audit report?",
        "evaluation_report": state.get("evaluation_report", {}),
        "evaluation_report_audit": state.get("evaluation_report_audit", {}),
    })
    approved = bool(approval.get("approved", False))
    reason = approval.get("reason", "")
    logger.info(f"[gate2] deployment_approved={approved} reason={reason!r}")
    return Command(
        goto="workflow_controller",
        update={
            "deployment_approved": approved,
            "deployment_decision": "approved" if approved else "rejected",
        },
    )
