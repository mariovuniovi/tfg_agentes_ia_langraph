"""Conflict detection tests for the Planner Agent's validation layer.

Hard conflicts block (raise PlannerValidationError if unaccompanied by a resolution).
Soft conflicts are informational (surfaced to the UI but never block).
"""
from __future__ import annotations

import pytest

from mlops_agents.contracts.planner import (
    DecisionBasis,
    EvidenceConflict,
    EvidenceReference,
    ExperienceSummary,
    PlannerOutput,
    PlannerValidationContext,
)
from mlops_agents.contracts.training import CandidateSpec, RejectedModelSpec, TrainingPlan
from mlops_agents.planning.trace import ToolTrace
from mlops_agents.planning.validation import (
    PlannerValidationError,
    _check_conflict_resolution_present_if_flagged,
    _detect_conflicts,
    detect_soft_conflicts,
)

# --- helpers ---

def _es(eid: str, best_model: str = "extra_trees_forecaster") -> ExperienceSummary:
    return ExperienceSummary(
        experience_id=eid,
        similarity_score=0.7,
        relevance_tier="high",
        matched_buckets=[],
        mismatched_buckets=[],
        target_scale_note=None,
        dataset_summary="",
        models_trained=[best_model],
        best_model=best_model,
        validation_score=0.5,
        metric_name="rmse",
        candidate_results=[],
    )


def _registry_ref(key: str) -> EvidenceReference:
    return EvidenceReference(source="registry", source_id=key)


def _exp_ref(eid: str) -> EvidenceReference:
    return EvidenceReference(source="experience", source_id=eid)


def _output(candidates, rejected=None, evidence_used=None, conflicts=None) -> PlannerOutput:
    return PlannerOutput(
        planning_analysis="ok",
        decision_basis=DecisionBasis(
            primary_evidence=[EvidenceReference(source="dataset_profile", source_id=None)],
            secondary_evidence=[],
            final_strategy="s",
        ),
        evidence_used=evidence_used or [],
        evidence_conflicts=conflicts or [],
        risks_or_warnings=[],
        plan=TrainingPlan(
            problem_type="forecasting",
            candidates=candidates,
            models_not_recommended=rejected or [],
        ),
    )


def _ctx(similar_experiences=None, matched_rules=None, rules_by_id=None) -> PlannerValidationContext:
    return PlannerValidationContext(
        problem_type="forecasting",
        task_metadata={},
        available_model_keys=["ets", "extra_trees_forecaster"],
        available_model_specs=[],
        similar_experiences=similar_experiences or [],
        matched_rules=matched_rules or [],
        rules_by_id=rules_by_id or {},
    )


# --- cited experience winner not selected is SOFT (expected under a capped set) ---

def test_cited_experience_winner_not_selected_is_soft():
    cand = CandidateSpec(
        model_key="ets", priority=1, reason="ok",
        evidence_refs=[_registry_ref("ets")],
    )
    out = _output(candidates=[cand], evidence_used=[_exp_ref("e1")])
    ctx = _ctx(similar_experiences=[_es("e1", best_model="extra_trees_forecaster")])
    trace = ToolTrace(retrieved_experience_ids=["e1"])
    # Not a hard/blocking conflict any more (the planner is capped to a bounded set,
    # so some cited winners naturally do not make the selection).
    hard = _detect_conflicts(ctx, trace, out.plan, out)
    assert not any(c["type"] == "cited_experience_winner_not_selected" for c in hard)
    # ...but still surfaced as a soft, informational signal.
    soft = detect_soft_conflicts(ctx, trace, out.plan, out)
    assert any(c["type"] == "cited_experience_winner_not_selected" for c in soft)


# --- soft conflict: retrieved but not cited ---

def test_soft_conflict_only_when_retrieved_but_not_cited():
    cand = CandidateSpec(
        model_key="ets", priority=1, reason="ok",
        evidence_refs=[_registry_ref("ets")],
    )
    out = _output(candidates=[cand])  # no cited experiences
    ctx = _ctx(similar_experiences=[_es("e1", best_model="extra_trees_forecaster")])
    trace = ToolTrace(retrieved_experience_ids=["e1"])

    soft = detect_soft_conflicts(ctx, trace, out.plan, out)
    assert any(c["type"] == "retrieved_experience_winner_not_selected" for c in soft)

    hard = _detect_conflicts(ctx, trace, out.plan, out)
    assert not any(c["type"] == "cited_experience_winner_not_selected" for c in hard)


# --- resolution required for flagged hard conflicts ---

def test_hard_conflict_resolution_required():
    # A cited rule whose avoided model is selected is still a HARD conflict; without a
    # resolution in evidence_conflicts it must raise.
    cand = CandidateSpec(
        model_key="extra_trees_forecaster", priority=1, reason="ok",
        evidence_refs=[_registry_ref("extra_trees_forecaster")],
    )
    rule_ref = EvidenceReference(source="rule", source_id="rule_short_history")
    out = _output(candidates=[cand], evidence_used=[rule_ref])
    ctx = _ctx(rules_by_id={
        "rule_short_history": {
            "rule_id": "rule_short_history",
            "prefer": ["ets"],
            "avoid_or_deprioritize": ["extra_trees_forecaster"],
            "recommend": [],
        }
    })
    trace = ToolTrace(retrieved_rule_ids=["rule_short_history"])

    with pytest.raises(PlannerValidationError, match="evidence_conflicts is empty"):
        _check_conflict_resolution_present_if_flagged(out, ctx, trace)


def test_hard_conflict_with_resolution_passes():
    cand = CandidateSpec(
        model_key="extra_trees_forecaster", priority=1, reason="ok",
        evidence_refs=[_registry_ref("extra_trees_forecaster")],
    )
    rule_ref = EvidenceReference(source="rule", source_id="rule_short_history")
    resolved = EvidenceConflict(
        summary="rule avoids extra_trees but exogenous features justify it",
        affected_models=["extra_trees_forecaster"],
        conflicting_evidence_refs=[rule_ref],
        resolution="exogenous calendar features present; tree model justified",
    )
    out = _output(
        candidates=[cand],
        evidence_used=[rule_ref],
        conflicts=[resolved],
    )
    ctx = _ctx(rules_by_id={
        "rule_short_history": {
            "rule_id": "rule_short_history",
            "prefer": ["ets"],
            "avoid_or_deprioritize": ["extra_trees_forecaster"],
            "recommend": [],
        }
    })
    trace = ToolTrace(retrieved_rule_ids=["rule_short_history"])

    _check_conflict_resolution_present_if_flagged(out, ctx, trace)  # no raise


# --- cited rule conflicts ---

def test_cited_rule_avoid_violated_flagged():
    cand = CandidateSpec(
        model_key="extra_trees_forecaster", priority=1, reason="ok",
        evidence_refs=[_registry_ref("extra_trees_forecaster")],
    )
    rule_ref = EvidenceReference(source="rule", source_id="rule_short_history")
    out = _output(candidates=[cand], evidence_used=[rule_ref])
    ctx = _ctx(rules_by_id={
        "rule_short_history": {
            "rule_id": "rule_short_history",
            "prefer": ["ets"],
            "avoid_or_deprioritize": ["extra_trees_forecaster"],
            "recommend": [],
        }
    })
    trace = ToolTrace(retrieved_rule_ids=["rule_short_history"])

    conflicts = _detect_conflicts(ctx, trace, out.plan, out)
    assert any(c["type"] == "cited_rule_avoid_violated" for c in conflicts)


def test_cited_rule_prefer_rejected_is_soft():
    # A cited rule preferring a model that the (capped) plan rejected is expected,
    # so it is a SOFT signal, not a hard/blocking conflict.
    cand = CandidateSpec(
        model_key="extra_trees_forecaster", priority=1, reason="ok",
        evidence_refs=[_registry_ref("extra_trees_forecaster")],
    )
    rej = RejectedModelSpec(
        model_key="ets", reason="too simple",
        evidence_refs=[_registry_ref("ets")],
    )
    rule_ref = EvidenceReference(source="rule", source_id="rule_short_history")
    out = _output(candidates=[cand], rejected=[rej], evidence_used=[rule_ref])
    ctx = _ctx(rules_by_id={
        "rule_short_history": {
            "rule_id": "rule_short_history",
            "prefer": ["ets"],
            "avoid_or_deprioritize": [],
            "recommend": [],
        }
    })
    trace = ToolTrace(retrieved_rule_ids=["rule_short_history"])

    hard = _detect_conflicts(ctx, trace, out.plan, out)
    assert not any(c["type"] == "cited_rule_prefer_rejected" for c in hard)
    soft = detect_soft_conflicts(ctx, trace, out.plan, out)
    assert any(c["type"] == "cited_rule_prefer_rejected" for c in soft)


# --- no false positives ---

def test_no_conflicts_on_clean_plan():
    cand = CandidateSpec(
        model_key="extra_trees_forecaster", priority=1, reason="ok",
        evidence_refs=[_registry_ref("extra_trees_forecaster")],
    )
    out = _output(candidates=[cand], evidence_used=[_exp_ref("e1")])
    ctx = _ctx(similar_experiences=[_es("e1", best_model="extra_trees_forecaster")])
    trace = ToolTrace(retrieved_experience_ids=["e1"])
    hard = _detect_conflicts(ctx, trace, out.plan, out)
    assert hard == []


def test_empty_resolution_raises():
    cand = CandidateSpec(
        model_key="ets", priority=1, reason="ok",
        evidence_refs=[_registry_ref("ets")],
    )
    bad = EvidenceConflict(
        summary="x", affected_models=["ets"],
        conflicting_evidence_refs=[_registry_ref("ets")],
        resolution="   ",  # whitespace-only
    )
    out = _output(candidates=[cand], conflicts=[bad])
    ctx = _ctx()
    trace = ToolTrace()
    with pytest.raises(PlannerValidationError, match="empty resolution"):
        _check_conflict_resolution_present_if_flagged(out, ctx, trace)
