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
    import pandas as pd
    counts = state.get("agent_attempt_counts") or {}
    attempt = counts.get("data_validator", 1)

    path = state.get("processed_dataset_path", "")
    preview: dict[str, Any] = {
        "path": path,
        "row_count": 0,
        "column_count": 0,
        "shape": [0, 0],
        "columns": [],
        "sample_rows": [],
        "head": [],
        "tail": [],
    }
    if path:
        try:
            df = pd.read_csv(path)
            head = df.head(5).to_dict(orient="records")
            tail = (
                df.tail(5).to_dict(orient="records")
                if state.get("problem_type") == "forecasting"
                else []
            )
            preview = {
                "path": path,
                "row_count": int(len(df)),
                "column_count": int(len(df.columns)),
                "shape": [int(len(df)), int(len(df.columns))],
                "columns": [
                    {"name": str(c), "dtype": str(df[c].dtype)} for c in df.columns
                ],
                "sample_rows": head,
                "head": head,
                "tail": tail,
            }
        except Exception as exc:
            logger.warning(f"[gate1] failed to build dataset_preview from {path}: {exc}")

    approval = interrupt({
        "type": "data_validation",
        "question": "Review the processed dataset before training begins.",
        "attempt": attempt,
        "dataset_preview": preview,
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
