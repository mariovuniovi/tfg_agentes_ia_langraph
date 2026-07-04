"""Evaluation Report LLM node — evidence-aligned audit, not a decision maker.

The promotion decision is already made by evaluate_promotion() before this
node runs. This module produces a structured narrative that triangulates
planner reasoning, training results, and empirical metrics.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class EvaluationReport(BaseModel):
    summary: str
    champion_model: str
    why_champion_won: str
    planner_alignment: str
    deviations_from_planner_expectations: list[str] = Field(default_factory=list)
    evidence_consistency_warnings: list[str] = Field(default_factory=list)
    risks_and_warnings: list[str] = Field(default_factory=list)
    promotion_decision_explanation: str
    human_review_notes: list[str] = Field(default_factory=list)


from functools import cache
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from mlops_agents.prompts import get_prompt
from mlops_agents.utils.llm import get_llm
from mlops_agents.utils.logging import get_logger

logger = get_logger(__name__)

_report_prompt = get_prompt("report_writer").template


def build_report_writer():
    """Return an LLM bound to the EvaluationReport structured output schema."""
    llm = get_llm("report_writer")
    return llm.with_structured_output(EvaluationReport, method="function_calling")


@cache
def get_report_writer_agent() -> Any:
    """Return the report writer LLM, built lazily on first use and cached."""
    return build_report_writer()


def _stub_report(reason: str) -> EvaluationReport:
    return EvaluationReport(
        summary=f"Audit report unavailable due to LLM error: {reason}",
        champion_model="",
        why_champion_won="",
        planner_alignment="",
        deviations_from_planner_expectations=[],
        evidence_consistency_warnings=["audit_unavailable"],
        risks_and_warnings=["audit_unavailable"],
        promotion_decision_explanation="See deterministic evaluation_report for the decision.",
        human_review_notes=[],
    )


def _build_audit_context(state: dict[str, Any]) -> HumanMessage:
    import json
    payload = {
        "evaluation_passed": state.get("evaluation_passed"),
        "candidate_metrics": state.get("candidate_metrics", {}),
        "champion_metrics": state.get("champion_metrics", {}),
        "thresholds_applied": state.get("thresholds_applied", {}),
        "planner_output_record": state.get("_planner_output_record", {}),
        "training_plan": state.get("training_plan", {}),
        "champion_candidate": state.get("champion_candidate", {}),
    }
    return HumanMessage(content=json.dumps(payload, default=str, indent=2))


def run_report_writer(state: dict[str, Any]) -> dict[str, Any]:
    """Invoke the audit LLM; retry once, then write a stub on persistent failure."""
    agent = get_report_writer_agent()
    ctx = _build_audit_context(state)
    messages = [SystemMessage(content=_report_prompt), ctx]

    last_err = ""
    for attempt in range(2):
        try:
            report: EvaluationReport = agent.invoke(messages)
            status = "ok" if attempt == 0 else "retry_ok"
            logger.info(f"[report_writer] status={status}")
            return {
                "evaluation_report_audit": report.model_dump(),
                "evaluation_report_audit_status": status,
            }
        except Exception as exc:
            last_err = str(exc)
            logger.warning(f"[report_writer] attempt {attempt + 1} failed: {last_err}")
            if attempt == 0:
                messages = list(messages) + [HumanMessage(
                    content=f"Previous attempt failed: {last_err}. Produce a valid EvaluationReport."
                )]

    stub = _stub_report(last_err)
    return {
        "evaluation_report_audit": stub.model_dump(),
        "evaluation_report_audit_status": "stub",
    }
