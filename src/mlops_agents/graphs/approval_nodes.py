"""HITL approval nodes — pure interrupt() wrappers.

Each node pauses the graph with interrupt() and writes an approval flag
to state. Payload `type` values are preserved from the old embedded HITL
so the existing SSE event shape stays backward compatible.
"""
from __future__ import annotations

from typing import Any

from langgraph.types import Command, interrupt

from mlops_agents.contracts.outputs import DatasetApprovalStateUpdate, DeploymentApprovalStateUpdate
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
        "join_plan": state.get("data_join_plan"),
        "join_evaluations": state.get("data_join_evaluations"),
        "join_base_nrows": state.get("data_join_base_nrows"),
    })
    approved = bool(approval.get("approved", False))
    comment = approval.get("comment", "")
    logger.info(f"[gate1] dataset_approved={approved} comment={comment!r}")
    return Command(
        goto="workflow_controller",
        update=DatasetApprovalStateUpdate(
            dataset_approved=approved,
            dataset_rejection_comment="" if approved else comment,
        ).to_update(),
    )


def deployment_approval_node(state: dict[str, Any]) -> Command:
    from mlops_agents.evaluation.champion import resolve_champion_model_name
    champion = resolve_champion_model_name(state)

    approval = interrupt({
        "type": "deployer",
        "question": "Approve deployment of this model based on the audit report?",
        "evaluation_report":       state.get("evaluation_report", {}),
        "evaluation_report_audit": state.get("evaluation_report_audit", {}),
        "candidate_metrics":       state.get("candidate_metrics", {}),
        "champion_metrics":        state.get("champion_metrics", {}),
        "thresholds_applied":      state.get("thresholds_applied", {}),
        "training_plan":           state.get("training_plan", {}),
        "candidate_run_id":        state.get("training_run_id", ""),
        "deployment_action": {
            "verb":    "register_and_promote",
            "model":   champion,
            "alias":   "champion",
            "summary": "This approval will register the candidate run as a new model version and assign it the champion alias.",
        },
    })
    approved = bool(approval.get("approved", False))
    reason = approval.get("reason", "")
    logger.info(f"[gate2] deployment_approved={approved} reason={reason!r} model={champion!r}")
    return Command(
        goto="workflow_controller",
        update=DeploymentApprovalStateUpdate(deployment_approved=approved).to_update(),
    )
