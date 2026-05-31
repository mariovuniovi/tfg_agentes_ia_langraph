"""Tests for CandidateSpec, RejectedModelSpec, and EvidenceReference contracts (SP5 Task 1.1).

NOTE on strictness: reason and evidence_refs on CandidateSpec/RejectedModelSpec default to ""
and [] respectively for backward compatibility with the 20+ existing constructors that predate
SP5. min_length enforcement is deferred to Task 5.2 (planner_node), which will validate the
planner's output specifically before it enters the graph state.
"""

import pytest
from pydantic import ValidationError

from mlops_agents.contracts.planner import (
    CandidateSpec,
    DecisionBasis,
    EvidenceConflict,
    EvidenceReference,
    ExperienceSummary,
    PlannerOutput,
    RejectedModelSpec,
)
from mlops_agents.contracts.training import TrainingPlan


def _ref(source="registry", source_id="lr"):
    return EvidenceReference(source=source, source_id=source_id, relevance_note="x")


def _minimal_training_plan():
    """Reusable helper — every test in this file that needs a TrainingPlan should call this."""
    return TrainingPlan(
        problem_type="regression",
        candidates=[CandidateSpec(model_key="ridge", priority=1, reason="ok", evidence_refs=[_ref()])],
        models_not_recommended=[],
    )


# --- EvidenceReference ---

def test_evidence_reference_with_relevance_note():
    ref = EvidenceReference(source="registry", source_id="lr", relevance_note="x")
    assert ref.relevance_note == "x"
    assert ref.summary == ""  # default


def test_evidence_reference_summary_optional():
    ref = EvidenceReference(source="experience", source_id="exp_001")
    assert ref.summary == ""
    assert ref.relevance_note is None


def test_evidence_reference_full_fields():
    ref = EvidenceReference(
        source="dataset_profile",
        source_id=None,
        summary="medium dataset",
        relevance_note="relevant for size-based rules",
    )
    assert ref.summary == "medium dataset"
    assert ref.relevance_note == "relevant for size-based rules"


# --- CandidateSpec ---

def test_candidate_requires_priority_ge_1():
    with pytest.raises(ValidationError):
        CandidateSpec(model_key="ridge", priority=0, reason="x", evidence_refs=[_ref()])


def test_candidate_valid_minimal_with_reason_and_evidence():
    spec = CandidateSpec(model_key="ridge", priority=1, reason="ok", evidence_refs=[_ref()])
    assert spec.risks == []
    assert spec.reason == "ok"
    assert len(spec.evidence_refs) == 1


def test_candidate_defaults_for_backward_compat():
    """Existing constructors without reason/evidence_refs must not break."""
    spec = CandidateSpec(model_key="ridge", priority=1)
    assert spec.reason == ""
    assert spec.evidence_refs == []
    assert spec.risks == []


def test_candidate_risks_default_empty_list():
    spec = CandidateSpec(model_key="ridge", priority=1, reason="ok", evidence_refs=[_ref()])
    assert spec.risks == []


def test_candidate_risks_populated():
    spec = CandidateSpec(
        model_key="ridge", priority=1, reason="ok",
        evidence_refs=[_ref()], risks=["may underfit", "high bias"],
    )
    assert spec.risks == ["may underfit", "high bias"]


def test_candidate_preserves_existing_fields():
    """Existing fields (initial_hyperparameters, search_space_override, etc.) still work."""
    spec = CandidateSpec(
        model_key="ridge", priority=2,
        initial_hyperparameters={"alpha": 1.0},
        requested_trials=10,
    )
    assert spec.initial_hyperparameters == {"alpha": 1.0}
    assert spec.requested_trials == 10


# --- RejectedModelSpec ---

def test_rejected_model_spec_basic():
    spec = RejectedModelSpec(model_key="lstm", reason="too complex")
    assert spec.model_key == "lstm"
    assert spec.reason == "too complex"
    assert spec.evidence_refs == []
    assert spec.reconsider_if is None


def test_rejected_accepts_optional_reconsider_if():
    spec = RejectedModelSpec(
        model_key="lstm", reason="too complex",
        evidence_refs=[_ref()],
        reconsider_if="more data becomes available",
    )
    assert spec.reconsider_if == "more data becomes available"


def test_rejected_evidence_refs_default_empty():
    """Backward-compat: constructors without evidence_refs must not break."""
    spec = RejectedModelSpec(model_key="lstm", reason="too slow")
    assert spec.evidence_refs == []


def test_rejected_with_evidence_refs():
    spec = RejectedModelSpec(
        model_key="lstm", reason="too complex", evidence_refs=[_ref(), _ref("experience", "exp_1")]
    )
    assert len(spec.evidence_refs) == 2


# --- Training plan integration ---

def test_minimal_training_plan_with_candidate_spec():
    plan = _minimal_training_plan()
    assert plan.candidates[0].model_key == "ridge"
    assert plan.candidates[0].priority == 1


def test_training_plan_backward_compat_no_reason():
    """Plans built without reason still work — backward compat preserved."""
    plan = TrainingPlan(
        problem_type="regression",
        candidates=[CandidateSpec(model_key="ridge", priority=1)],
        models_not_recommended=[RejectedModelSpec(model_key="lightgbm_regressor", reason="overkill")],
    )
    assert plan.candidates[0].reason == ""
    assert plan.models_not_recommended[0].reconsider_if is None


# --- Aliases (backward compat) ---

def test_training_plan_candidate_alias():
    from mlops_agents.contracts.training import TrainingPlanCandidate, CandidateSpec
    assert TrainingPlanCandidate is CandidateSpec


def test_rejected_model_alias():
    from mlops_agents.contracts.training import RejectedModel, RejectedModelSpec
    assert RejectedModel is RejectedModelSpec


# --- DecisionBasis ---

def test_decision_basis_requires_primary_evidence():
    with pytest.raises(ValidationError):
        DecisionBasis(primary_evidence=[], secondary_evidence=[], final_strategy="x")


def test_decision_basis_requires_non_empty_final_strategy():
    with pytest.raises(ValidationError):
        DecisionBasis(primary_evidence=[_ref()], secondary_evidence=[], final_strategy="")


# --- EvidenceConflict ---

def test_evidence_conflict_requires_resolution():
    with pytest.raises(ValidationError):
        EvidenceConflict(
            summary="x", affected_models=["lr"],
            conflicting_evidence_refs=[_ref()], resolution="",
        )


# --- ExperienceSummary extension ---

def test_experience_summary_has_relevance_tier_buckets_scale_note():
    es = ExperienceSummary(
        experience_id="e1",
        similarity_score=0.72,
        relevance_tier="high",
        matched_buckets=["forecasting", "weekly"],
        mismatched_buckets=["target_scale"],
        target_scale_note="candidate target std ≈10× experience",
        dataset_summary="s",
        models_trained=["ets"],
        best_model="ets",
        validation_score=0.5,
        metric_name="rmse",
        candidate_results=[],
    )
    assert es.relevance_tier == "high"


# --- PlannerOutput extension ---

def test_planner_output_has_decision_basis_and_evidence_conflicts():
    out = PlannerOutput(
        planning_analysis="ok",
        decision_basis=DecisionBasis(
            primary_evidence=[_ref()],
            secondary_evidence=[],
            final_strategy="prioritize simple models",
        ),
        evidence_used=[],
        evidence_conflicts=[],
        risks_or_warnings=[],
        plan=_minimal_training_plan(),
    )
    assert out.decision_basis.final_strategy == "prioritize simple models"
    assert out.evidence_conflicts == []


# --- PlannerValidationContext ---

def test_planner_validation_context_minimal():
    from mlops_agents.contracts.planner import PlannerValidationContext
    from mlops_agents.models.loader import get_models_for

    specs = get_models_for("forecasting")
    ctx = PlannerValidationContext(
        problem_type="forecasting",
        task_metadata={},
        available_model_keys=[s.model_key for s in specs],
        available_model_specs=specs,
        similar_experiences=[],
        matched_rules=[],
        rules_by_id={},
    )
    assert ctx.problem_type == "forecasting"
    assert len(ctx.available_model_keys) > 0
