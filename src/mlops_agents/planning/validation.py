"""Validation chain for the Planner Agent's PlannerOutput.

Hybrid validation (A1):
- Hard invariants (registry, exhaustiveness, plan integrity) → deterministic context
- Citation honesty (experience + rule refs) → agent's observed ToolTrace
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from mlops_agents.config.settings import settings
from mlops_agents.contracts.planner import (
    EvidenceReference,
    PlannerOutput,
)
from mlops_agents.contracts.training import (
    ForecastingSettings,
    PlannerTrainingPlan,
    TrainingPlan,
)
from mlops_agents.planning.context import PlannerValidationContext
from mlops_agents.planning.trace import ToolTrace

REQUIRED_TOOLS = {
    "list_available_models",
    "retrieve_similar_experiences",
    "retrieve_ml_knowledge",
}
ALLOWED_VAL_STRATEGIES = {"single_split", "rolling_window", "expanding_window"}
ALLOWED_EXOG_STRATEGIES = {"naive_carry", "ets", "auto_arima"}


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

    # 5. forecasting_settings is code-resolved, not part of the LLM's decision
    #    (PlannerTrainingPlan has no such field). The resolved settings are
    #    validated separately by validate_forecasting_settings(), called from
    #    planner_node against the deterministically-built forecasting_fs.

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


def validate_forecasting_settings(fs: ForecastingSettings, task_metadata: dict[str, Any]) -> None:
    """Defense-in-depth check on the *code-resolved* forecasting settings.

    These come from resolve_validation_strategy / resolve_exog_strategies (not from
    the LLM), so this guards against resolver bugs, not bad agent output. Called by
    planner_node against the deterministically-built forecasting_fs. Raises
    PlannerValidationError on an invalid setting.
    """
    val_strat = getattr(fs.validation_strategy, "type", None)
    if val_strat not in ALLOWED_VAL_STRATEGIES:
        raise PlannerValidationError(
            f"invalid validation_strategy: {val_strat!r}. "
            f"Allowed: {sorted(ALLOWED_VAL_STRATEGIES)}"
        )
    exog_strats = getattr(fs.exog_strategies, "per_column", {}) or {}
    # Derive known-future columns from the canonical metadata shape used everywhere
    # else (exog_policy.resolve_exog_strategies, forecasting_runner.resolve_exog_availability):
    # exogenous_columns: [{"name": ..., "future_availability": "known_future" | ...}].
    known_future = {
        e["name"]
        for e in (task_metadata.get("exogenous_columns") or [])
        if e.get("future_availability") == "known_future"
    }
    for col, strat in exog_strats.items():
        if strat not in ALLOWED_EXOG_STRATEGIES:
            raise PlannerValidationError(
                f"invalid exog strategy {strat!r} for column {col!r}. "
                f"Allowed: {sorted(ALLOWED_EXOG_STRATEGIES)}"
            )
        if col in known_future:
            raise PlannerValidationError(
                f"known_future column {col!r} cannot appear in per-column "
                f"unknown-future strategies"
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


def _detect_conflicts(
    ctx: PlannerValidationContext,
    trace: ToolTrace,
    plan: PlannerTrainingPlan,
    output: PlannerOutput,
) -> list[dict[str, Any]]:
    """Deterministic HARD conflict detection. Returns list of flagged conflicts."""
    hard: list[dict[str, Any]] = []
    candidate_keys = {c.model_key for c in plan.candidates}

    # NOTE: "cited experience winner not selected" is intentionally NOT a hard conflict.
    # The planner is capped to a bounded candidate set (top-5), so it is expected that
    # some winners of cited experiences do not make the selection. Treating that as a
    # hard conflict forced needless retries (and occasional planner failures); it is now
    # surfaced as a SOFT, non-blocking signal in detect_soft_conflicts() instead.
    cited_rule_ids = {
        ref.source_id for ref in _collect_all_refs(output)
        if ref.source == "rule" and ref.source_id
    }
    for rid in cited_rule_ids:
        rule = ctx.rules_by_id.get(rid)
        if not rule:
            continue
        avoid_in_cands = set(rule.get("avoid_or_deprioritize", []) or []) & candidate_keys
        if avoid_in_cands:
            hard.append({
                "type": "cited_rule_avoid_violated", "rule_id": rid,
                "models": sorted(avoid_in_cands), "severity": "hard",
            })
        # NOTE: cited_rule_prefer_rejected (a cited rule prefers a model that ended up
        # rejected) is NOT hard: under a capped candidate set the planner must reject
        # some rule-preferred models. It is surfaced as soft in detect_soft_conflicts().
    return hard


def detect_soft_conflicts(
    ctx: PlannerValidationContext,
    trace: ToolTrace,
    plan: TrainingPlan,
    output: PlannerOutput,
) -> list[dict[str, Any]]:
    """Non-blocking conflicts surfaced as info in the UI. Excludes anything already in hard."""
    soft: list[dict[str, Any]] = []
    candidate_keys = {c.model_key for c in plan.candidates}

    retrieved_winners = {
        e.best_model for e in ctx.similar_experiences
        if e.experience_id in trace.retrieved_experience_ids and e.best_model
    }
    cited_experience_ids = {
        ref.source_id for ref in _collect_all_refs(output)
        if ref.source == "experience" and ref.source_id
    }
    cited_winners = {
        e.best_model for e in ctx.similar_experiences
        if e.experience_id in cited_experience_ids and e.best_model
    }
    soft_omitted = (retrieved_winners - cited_winners) - candidate_keys
    if soft_omitted:
        soft.append({
            "type": "retrieved_experience_winner_not_selected",
            "models": sorted(soft_omitted),
            "summary": (
                f"{len(soft_omitted)} model(s) won in retrieved experiences but were not "
                f"cited or selected: {sorted(soft_omitted)}."
            ),
        })

    cited_omitted = cited_winners - candidate_keys
    if cited_omitted:
        soft.append({
            "type": "cited_experience_winner_not_selected",
            "models": sorted(cited_omitted),
            "summary": (
                f"{len(cited_omitted)} model(s) won in a cited experience but were not "
                f"selected (expected under a capped candidate set): {sorted(cited_omitted)}."
            ),
        })

    rejected_keys = {r.model_key for r in plan.models_not_recommended}
    cited_rule_ids = {
        ref.source_id for ref in _collect_all_refs(output)
        if ref.source == "rule" and ref.source_id
    }
    for rid in cited_rule_ids:
        rule = ctx.rules_by_id.get(rid)
        if not rule:
            continue
        prefer_in_rej = set(rule.get("prefer", []) or []) & rejected_keys
        if prefer_in_rej:
            soft.append({
                "type": "cited_rule_prefer_rejected",
                "rule_id": rid,
                "models": sorted(prefer_in_rej),
                "summary": (
                    f"cited rule {rid!r} prefers {sorted(prefer_in_rej)} but they were "
                    f"rejected (expected under a capped candidate set)."
                ),
            })
    return soft


def _check_conflict_resolution_present_if_flagged(
    output: PlannerOutput,
    ctx: PlannerValidationContext,
    trace: ToolTrace,
) -> None:
    flagged = _detect_conflicts(ctx, trace, output.plan, output)
    if flagged and not output.evidence_conflicts:
        raise PlannerValidationError(
            f"deterministic conflict detector flagged {len(flagged)} conflict(s) but "
            f"evidence_conflicts is empty. Flagged: {flagged}"
        )
    for c in output.evidence_conflicts:
        if not c.resolution.strip():
            raise PlannerValidationError(
                f"evidence_conflict for {c.affected_models} has empty resolution"
            )
