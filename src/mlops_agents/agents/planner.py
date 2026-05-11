"""Model Planning Agent — context builder, validators, and planner node (SP5)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command
from pydantic import ValidationError

from mlops_agents.config.settings import settings
from mlops_agents.contracts.planner import (
    CandidateResultCompact,
    EvidenceReference,
    ExperienceSummary,
    PlannerContext,
    PlannerOutput,
)
from mlops_agents.contracts.training import TrainingPlan
from mlops_agents.experience.pool import ExperiencePool
from mlops_agents.experience.schema import RetrievalView
from mlops_agents.knowledge.reader import match_rules
from mlops_agents.models.loader import get_models_for
from mlops_agents.prompts import get_prompt
from mlops_agents.state.agent_state import AgentState
from mlops_agents.training.profiler import build_dataset_profile
from mlops_agents.utils.llm import get_llm
from mlops_agents.utils.logging import get_logger

logger = get_logger(__name__)


class PlannerError(Exception):
    """Raised when the planner fails validation after all retry attempts."""


def _to_experience_summary(view: RetrievalView) -> ExperienceSummary:
    """Convert a RetrievalView to the compact ExperienceSummary sent to the LLM."""
    sel_key = view.selected_solution.model_key
    scored = [c for c in view.models_tested if c.best_score is not None]
    failed = [c for c in view.models_tested if c.best_score is None]

    # Champion first, then remaining by score descending
    scored.sort(key=lambda c: (c.model_key != sel_key, -(c.best_score or 0.0)))
    compact = [
        CandidateResultCompact(model_key=c.model_key, rank=i + 1, metric_value=c.best_score)
        for i, c in enumerate(scored)
    ]
    for f in failed:
        compact.append(CandidateResultCompact(
            model_key=f.model_key, rank=len(compact) + 1, metric_value=None
        ))

    return ExperienceSummary(
        experience_id=view.task_id,
        similarity_score=view.similarity_ratio,
        dataset_summary=view.experience_summary or "",
        models_trained=[c.model_key for c in view.models_tested],
        best_model=sel_key,
        validation_score=view.selected_solution.validation_score,
        candidate_results=compact,
    )


def build_planner_context(
    profile: dict[str, Any],
    task_metadata: dict[str, Any],
    problem_type: str,
    pool: ExperiencePool,
    k: int = 5,
) -> PlannerContext:
    """Assemble PlannerContext deterministically — no LLM calls."""
    available_models = [m.model_key for m in get_models_for(problem_type)]
    views = pool.find_similar(profile, problem_type, k)
    similar_experiences = [_to_experience_summary(v) for v in views]
    rule_input = {**profile, **task_metadata, "problem_type": problem_type}
    matched = match_rules(rule_input)
    matched_rules_dicts: list[dict[str, Any]] = [
        {
            "rule_id": r.rule_id,
            "prefer": r.prefer,
            "avoid_or_deprioritize": r.avoid_or_deprioritize,
            "recommend": r.recommend,
            "summary": r.reason,
        }
        for r in matched
    ]
    return PlannerContext(
        current_dataset_profile=profile,
        task_metadata=task_metadata,
        available_models=available_models,
        similar_experiences=similar_experiences,
        matched_rules=matched_rules_dicts,
    )


def _check_evidence_references(
    refs: list[EvidenceReference], ctx: PlannerContext
) -> None:
    """Verify every EvidenceReference source_id exists in the context. Raises PlannerError."""
    exp_ids = {e.experience_id for e in ctx.similar_experiences}
    rule_ids = {r["rule_id"] for r in ctx.matched_rules}
    model_keys = set(ctx.available_models)

    for ref in refs:
        if ref.source in ("dataset_profile", "task_metadata"):
            if ref.source_id is not None:
                raise PlannerError(
                    f"{ref.source} reference must have source_id=None, got {ref.source_id!r}"
                )
        elif ref.source == "experience":
            if ref.source_id not in exp_ids:
                raise PlannerError(
                    f"experience source_id {ref.source_id!r} not in context"
                )
        elif ref.source == "rule":
            if ref.source_id not in rule_ids:
                raise PlannerError(
                    f"rule source_id {ref.source_id!r} not in context"
                )
        elif ref.source == "registry" and ref.source_id is not None and ref.source_id not in model_keys:
            raise PlannerError(
                f"registry source_id {ref.source_id!r} not in available_models"
            )


def _check_plan_exhaustiveness(
    plan: TrainingPlan, available_models: list[str]
) -> None:
    """Every model in available_models must appear in candidates or models_not_recommended."""
    accounted = (
        {c.model_key for c in plan.candidates}
        | {r.model_key for r in plan.models_not_recommended}
    )
    missing = set(available_models) - accounted
    if missing:
        raise PlannerError(
            f"These models are neither in candidates nor models_not_recommended: "
            f"{sorted(missing)}. Every available model must be explicitly included or rejected."
        )


_planner_prompt = get_prompt("planner").template


def planner_node(state: AgentState) -> Command[Literal["supervisor"]]:
    """Model Planning Agent node — assembles context, calls LLM, validates plan."""
    processed_path = Path(state["processed_dataset_path"])
    problem_type: str = state.get("problem_type", "")
    task_meta: dict[str, Any] = state.get("task_metadata") or {}

    # Reuse dataset_profile from state (produced by data_validator) to avoid recomputing.
    # Fall back to building it from the CSV only if the validator didn't store it.
    profile_raw = state.get("schema_json")
    if profile_raw:
        raw_dict = (
            profile_raw if isinstance(profile_raw, dict) else json.loads(profile_raw)
        )
    else:
        # build_dataset_profile requires "problem_type" in task_metadata
        profiler_meta = {**task_meta, "problem_type": problem_type}
        raw_dict = build_dataset_profile(processed_path, profiler_meta).model_dump()
    # PlannerContext.current_dataset_profile is typed as dict[str, str|int|float|bool]
    # — strip None values so Pydantic validation doesn't reject them.
    profile_dict = {k: v for k, v in raw_dict.items() if v is not None}

    pool = ExperiencePool(settings.experience_db_path)
    ctx = build_planner_context(profile_dict, task_meta, problem_type, pool)

    llm = get_llm("planner").with_structured_output(PlannerOutput)
    last_error = ""
    output: PlannerOutput

    for attempt in range(2):
        try:
            messages: list[SystemMessage | HumanMessage] = [
                SystemMessage(content=_planner_prompt),
                HumanMessage(content=ctx.model_dump_json(indent=2)),
            ]
            if attempt == 1:
                messages.append(HumanMessage(
                    content=f"Your previous plan was rejected: {last_error}. "
                            "Please produce a corrected PlannerOutput."
                ))
            output = cast(PlannerOutput, llm.invoke(messages))
            # Stage 3: evidence references
            _check_evidence_references(output.evidence_used, ctx)
            # Stage 4: exhaustiveness
            _check_plan_exhaustiveness(output.plan, ctx.available_models)
            break
        except (ValidationError, PlannerError, ValueError) as exc:
            last_error = str(exc)
            logger.warning(f"[planner] attempt {attempt + 1} failed: {last_error}")
            if attempt == 1:
                raise PlannerError(f"Planner failed after retry: {last_error}") from exc

    retry_used = attempt == 1

    planner_output_record = {
        "planner_status": "retry_ok" if retry_used else "ok",
        "retry_used": retry_used,
        "planning_analysis": output.planning_analysis,
        "evidence_used": [e.model_dump() for e in output.evidence_used],
        "risks_or_warnings": output.risks_or_warnings,
        "validation_errors": [last_error] if retry_used else [],
        "plan_summary": {
            "candidate_models": [c.model_key for c in output.plan.candidates],
            "models_not_recommended": [r.model_key for r in output.plan.models_not_recommended],
        },
        "prompt_version": "model_planner_v1",
    }

    logger.info(
        f"[planner] status={'retry_ok' if retry_used else 'ok'} "
        f"candidates={len(output.plan.candidates)} "
        f"rejected={len(output.plan.models_not_recommended)}"
    )

    return Command(
        goto="supervisor",
        update={
            "planner_analysis": output.planning_analysis,
            "planner_evidence_used": [e.model_dump() for e in output.evidence_used],
            "planner_warnings": output.risks_or_warnings,
            "planner_status": "retry_ok" if retry_used else "ok",
            "planner_retry_used": retry_used,
            "training_plan": output.plan.model_dump(),
            "_planner_output_record": planner_output_record,
        },
    )
