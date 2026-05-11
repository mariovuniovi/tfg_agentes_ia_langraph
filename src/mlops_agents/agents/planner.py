"""Model Planning Agent — context builder, validators, and planner node (SP5)."""
from __future__ import annotations
from typing import Any
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
    matched_rules_dicts = [
        {
            "rule_id": r.rule_id,
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
        elif ref.source == "registry":
            if ref.source_id is not None and ref.source_id not in model_keys:
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
