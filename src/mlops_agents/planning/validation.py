"""Validation chain for the Planner Agent's PlannerOutput.

Hybrid validation (A1):
- Hard invariants (registry, exhaustiveness, plan integrity) → deterministic context
- Citation honesty (experience + rule refs) → agent's observed ToolTrace
"""
from __future__ import annotations
from typing import Iterable

from mlops_agents.config.settings import settings
from mlops_agents.contracts.planner import (
    EvidenceReference, PlannerOutput, PlannerValidationContext,
)
from mlops_agents.contracts.training import TrainingPlan
from mlops_agents.planning.trace import ToolTrace


REQUIRED_TOOLS = {
    "list_available_models",
    "retrieve_similar_experiences",
    "retrieve_ml_knowledge",
}
ALLOWED_VAL_STRATEGIES = {"single_split", "rolling_window", "expanding_window"}
ALLOWED_EXOG_STRATEGIES = {"naive_carry", "ets", "auto_arima", "drop"}


class PlannerValidationError(Exception):
    """Raised when validation rejects a PlannerOutput. Caught by planner_node for retry."""


def _collect_all_refs(output: PlannerOutput) -> list[EvidenceReference]:
    """Union of every EvidenceReference appearing anywhere in PlannerOutput."""
    refs: list[EvidenceReference] = []
    refs.extend(output.evidence_used)
    if output.decision_basis is not None:
        refs.extend(output.decision_basis.primary_evidence)
        refs.extend(output.decision_basis.secondary_evidence)
    for c in output.plan.candidates:
        refs.extend(c.evidence_refs)
    for r in output.plan.models_not_recommended:
        refs.extend(r.evidence_refs)
    for conflict in output.evidence_conflicts:
        refs.extend(conflict.conflicting_evidence_refs)
    return refs


def _has_registry_self_ref(refs: Iterable[EvidenceReference], model_key: str) -> bool:
    return any(r.source == "registry" and r.source_id == model_key for r in refs)


def _check_plan_integrity(
    output: PlannerOutput,
    trace: ToolTrace,
    ctx: PlannerValidationContext,
) -> None:
    """All non-citation invariants. Raises PlannerValidationError on first failure."""
    # 1. Required tools
    missing = REQUIRED_TOOLS - set(trace.called_tools)
    if missing:
        raise PlannerValidationError(f"agent skipped required tools: {sorted(missing)}")

    # 2. Tool-call budget — global + per-tool inspect cap (defense-in-depth: tools
    # already reject calls past the cap, so this should never fire in practice).
    if trace.tool_call_count > settings.planner_max_tool_calls:
        raise PlannerValidationError(
            f"agent exceeded planner_max_tool_calls: {trace.tool_call_count} > "
            f"{settings.planner_max_tool_calls}"
        )
    if trace.inspect_model_details_count > settings.planner_max_inspect_calls:
        raise PlannerValidationError(
            f"agent exceeded planner_max_inspect_calls: "
            f"{trace.inspect_model_details_count} > {settings.planner_max_inspect_calls}"
        )

    # 3. Priority uniqueness + monotonicity
    priorities = [c.priority for c in output.plan.candidates]
    if any(p < 1 for p in priorities):
        raise PlannerValidationError(f"candidate priorities must be >= 1, got {priorities}")
    if len(set(priorities)) != len(priorities):
        raise PlannerValidationError(f"candidate priorities must be unique, got {priorities}")

    # 4. No candidate↔rejected overlap
    cand_set = {c.model_key for c in output.plan.candidates}
    rej_set = {r.model_key for r in output.plan.models_not_recommended}
    overlap = cand_set & rej_set
    if overlap:
        raise PlannerValidationError(
            f"models overlap candidates and rejected: {sorted(overlap)}"
        )

    # 5. Forecasting-specific settings
    if ctx.problem_type == "forecasting":
        fc = getattr(output.plan, "forecasting_settings", None)
        if fc is None:
            raise PlannerValidationError("forecasting plan missing forecasting_settings")
        val_strat = getattr(fc.validation_strategy, "type", None)
        if val_strat not in ALLOWED_VAL_STRATEGIES:
            raise PlannerValidationError(
                f"invalid validation_strategy: {val_strat!r}. "
                f"Allowed: {sorted(ALLOWED_VAL_STRATEGIES)}"
            )
        exog_strats = getattr(fc.exog_strategies, "per_column", {}) or {}
        known_future = set(ctx.task_metadata.get("known_future_columns", []))
        for col, strat in exog_strats.items():
            if strat not in ALLOWED_EXOG_STRATEGIES:
                raise PlannerValidationError(
                    f"invalid exog strategy {strat!r} for column {col!r}. "
                    f"Allowed: {sorted(ALLOWED_EXOG_STRATEGIES)}"
                )
            if col in known_future:
                raise PlannerValidationError(
                    f"known_future column {col!r} cannot appear in per-column "
                    f"unknown-future strategies (no 'drop' loophole)"
                )

    # 6. Per-candidate registry self-citation
    for c in output.plan.candidates:
        if not _has_registry_self_ref(c.evidence_refs, c.model_key):
            raise PlannerValidationError(
                f"candidate {c.model_key!r} missing registry self-citation "
                f"(evidence_refs must include source=registry, source_id={c.model_key!r})"
            )
    for r in output.plan.models_not_recommended:
        if not _has_registry_self_ref(r.evidence_refs, r.model_key):
            raise PlannerValidationError(
                f"rejected model {r.model_key!r} missing registry self-citation"
            )


def _check_plan_exhaustiveness(plan: TrainingPlan, available_model_keys: list[str]) -> None:
    accounted = (
        {c.model_key for c in plan.candidates}
        | {r.model_key for r in plan.models_not_recommended}
    )
    missing = set(available_model_keys) - accounted
    if missing:
        raise PlannerValidationError(
            f"models not classified as either candidates or rejected: {sorted(missing)}. "
            f"Every available model must be explicitly included."
        )


def _check_evidence_references_hybrid(
    output: PlannerOutput,
    ctx: PlannerValidationContext,
    trace: ToolTrace,
) -> None:
    """Hybrid: registry refs validated against deterministic context;
    experience/rule refs validated against agent's observed ToolTrace."""
    for ref in _collect_all_refs(output):
        if ref.source in ("dataset_profile", "task_metadata"):
            if ref.source_id is not None:
                raise PlannerValidationError(
                    f"{ref.source} ref must have source_id=None, got {ref.source_id!r}"
                )
        elif ref.source == "registry":
            if not ref.source_id:
                raise PlannerValidationError("registry ref requires non-empty source_id (model_key)")
            if ref.source_id not in ctx.available_model_keys:
                raise PlannerValidationError(
                    f"registry ref {ref.source_id!r} not in deterministic registry"
                )
        elif ref.source == "experience":
            if not ref.source_id:
                raise PlannerValidationError("experience ref requires non-empty source_id")
            if ref.source_id not in trace.retrieved_experience_ids:
                raise PlannerValidationError(
                    f"experience ref {ref.source_id!r} was never retrieved by the agent"
                )
        elif ref.source == "rule":
            if not ref.source_id:
                raise PlannerValidationError("rule ref requires non-empty source_id")
            if ref.source_id not in trace.retrieved_rule_ids:
                raise PlannerValidationError(
                    f"rule ref {ref.source_id!r} was never retrieved by the agent"
                )
