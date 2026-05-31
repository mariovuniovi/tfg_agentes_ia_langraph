"""planner_node — entry point, retry orchestration, validation."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command

from mlops_agents.config.settings import settings
from mlops_agents.planning.agent import build_planner_agent
from mlops_agents.planning.context import build_planner_validation_context
from mlops_agents.planning.prompts import build_retry_message, format_planner_inputs
from mlops_agents.planning.tools import build_planner_tools
from mlops_agents.planning.trace import ToolTrace
from mlops_agents.planning.validation import (
    PlannerValidationError,
    _check_conflict_resolution_present_if_flagged,
    _check_evidence_references_hybrid,
    _check_plan_exhaustiveness,
    _check_plan_integrity,
    _collect_all_refs,
    detect_soft_conflicts,
)
from mlops_agents.prompts import get_prompt
from mlops_agents.state.agent_state import AgentState
from mlops_agents.training.profiler import build_dataset_profile
from mlops_agents.utils.logging import get_logger

logger = get_logger(__name__)


class PlannerError(Exception):
    """Raised when the planner agent fails after the retry attempt."""


def planner_node(state: AgentState) -> Command[Literal["workflow_controller"]]:
    """Entry: build profile + validation context once, run agent up to 2 attempts."""
    processed_path = Path(state["processed_dataset_path"])
    problem_type: str = state.get("problem_type", "")  # type: ignore[union-attr]
    task_meta: dict[str, Any] = state.get("task_metadata") or {}  # type: ignore[union-attr]

    raw_profile = build_dataset_profile(
        processed_path, {**task_meta, "problem_type": problem_type}
    )
    profile_dict = raw_profile.model_dump()
    profile = {k: v for k, v in profile_dict.items() if v is not None}

    validation_ctx = build_planner_validation_context(profile, task_meta, problem_type)
    system_prompt = get_prompt("planner").template

    output = None
    trace = ToolTrace()
    last_error = ""
    retry_used = False

    for attempt in range(2):
        trace = ToolTrace()
        tools = build_planner_tools(profile, task_meta, problem_type, trace)
        agent = build_planner_agent(tools)

        messages: list = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=format_planner_inputs(profile, task_meta, problem_type)),
        ]
        if attempt == 1:
            messages.append(build_retry_message(last_error))
            retry_used = True

        try:
            result = agent.invoke(
                {"messages": messages},
                config={"recursion_limit": settings.planner_max_iterations},
            )
            output = result.get("structured_response")
            if output is None:
                raise PlannerValidationError(
                    "agent returned no structured_response — model failed to produce PlannerOutput"
                )

            _check_plan_integrity(output, trace, validation_ctx)
            _check_plan_exhaustiveness(output.plan, validation_ctx.available_model_keys)
            _check_evidence_references_hybrid(output, validation_ctx, trace)
            _check_conflict_resolution_present_if_flagged(output, validation_ctx, trace)
            break  # success
        except (PlannerValidationError, ValueError) as exc:
            last_error = str(exc)
            logger.warning(f"[planner] attempt {attempt + 1} failed: {last_error}")
            if attempt == 1:
                raise PlannerError(f"Planner failed after retry: {last_error}") from exc

    # Sort candidates by priority (deterministic order for executor)
    assert output is not None
    sorted_candidates = sorted(output.plan.candidates, key=lambda c: c.priority)
    output.plan.candidates = sorted_candidates

    soft = detect_soft_conflicts(validation_ctx, trace, output.plan, output)
    cited_experience_ids = sorted({
        r.source_id for r in _collect_refs_for_record(output)
        if r.source == "experience" and r.source_id
    })
    cited_rule_ids = sorted({
        r.source_id for r in _collect_refs_for_record(output)
        if r.source == "rule" and r.source_id
    })

    planner_status = "retry_ok" if retry_used else "ok"
    record = _build_planner_output_record(
        output, trace, validation_ctx, soft,
        cited_experience_ids, cited_rule_ids, planner_status, last_error,
    )

    logger.info(
        f"[planner] status={planner_status} candidates={len(output.plan.candidates)} "
        f"rejected={len(output.plan.models_not_recommended)} tool_calls={trace.tool_call_count}"
    )

    return Command(
        goto="workflow_controller",
        update={
            "planner_analysis": output.planning_analysis,
            "planner_evidence_used": [e.model_dump() for e in output.evidence_used],
            "planner_warnings": output.risks_or_warnings,
            "planner_status": planner_status,
            "planner_retry_used": retry_used,
            "training_plan": output.plan.model_dump(),
            "planner_tool_trace": trace.model_dump(),
            "planner_validation_context": _audit_subset(validation_ctx),
            "_planner_output_record": record,
        },
    )


# Helpers — placed below so test file imports cleanly

def _collect_refs_for_record(output: Any) -> list:
    return _collect_all_refs(output)


def _audit_subset(ctx: Any) -> dict:
    """Compact, JSON-serializable subset of validation context for state/audit."""
    return {
        "problem_type": ctx.problem_type,
        "available_model_keys": list(ctx.available_model_keys),
        "matched_rule_ids": [r["rule_id"] for r in ctx.matched_rules],
        "similar_experience_ids": [e.experience_id for e in ctx.similar_experiences],
    }


def _build_planner_output_record(
    output: Any,
    trace: ToolTrace,
    validation_ctx: Any,
    soft_conflicts: list,
    cited_experience_ids: list,
    cited_rule_ids: list,
    status: str,
    last_error: str,
) -> dict:
    """Compose the rich record consumed by the SSE pipeline & frontend PlannerPanel."""
    return {
        "planner_status": status,
        "retry_used": status == "retry_ok",
        "planning_analysis": output.planning_analysis,
        "decision_basis": output.decision_basis.model_dump() if output.decision_basis else None,
        "evidence_used": [e.model_dump() for e in output.evidence_used],
        "evidence_conflicts": [c.model_dump() for c in output.evidence_conflicts],
        "soft_conflicts": soft_conflicts,
        "risks_or_warnings": output.risks_or_warnings,
        "validation_errors": [last_error] if status == "retry_ok" else [],
        "plan_summary": {
            "candidate_rationales": [c.model_dump() for c in output.plan.candidates],
            "rejected_model_rationales": [r.model_dump() for r in output.plan.models_not_recommended],
            "candidate_models": [c.model_key for c in output.plan.candidates],
            "models_not_recommended": [r.model_key for r in output.plan.models_not_recommended],
        },
        "cited_experience_ids": cited_experience_ids,
        "cited_rule_ids": cited_rule_ids,
        "tool_trace": trace.model_dump(),
        "retrieved_experiences": [e.model_dump() for e in validation_ctx.similar_experiences],
        "matched_rules": validation_ctx.matched_rules,
        "prompt_version": "model_planner_v2",
    }
