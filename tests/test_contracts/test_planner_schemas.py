import pytest
from pydantic import ValidationError
from mlops_agents.contracts.planner import (
    EvidenceReference, CandidateSpec, RejectedModelSpec,
)
from mlops_agents.contracts.training import TrainingPlan, TrainingPlanCandidate


def _ref(source="registry", source_id="lr"):
    return EvidenceReference(source=source, source_id=source_id, relevance_note="x")


def _minimal_training_plan():
    """Reusable helper — every test in this file that needs a TrainingPlan should call
    this rather than inline the construction. If TrainingPlan's signature differs from
    what's shown here (field names, required fields, forecasting_settings shape), this
    is the ONE place to fix it — adjust to match the real schema in src/mlops_agents/contracts/training.py."""
    return TrainingPlan(
        problem_type="regression",
        candidates=[TrainingPlanCandidate(priority=1, model_key="ridge", reason="ok")],
        models_not_recommended=[],
    )

def test_candidate_requires_priority_ge_1():
    with pytest.raises(ValidationError):
        CandidateSpec(model_key="lr", priority=0, reason="x", evidence_refs=[_ref()], risks=[])

def test_candidate_requires_non_empty_evidence_refs():
    with pytest.raises(ValidationError):
        CandidateSpec(model_key="lr", priority=1, reason="x", evidence_refs=[], risks=[])

def test_candidate_requires_non_empty_reason():
    with pytest.raises(ValidationError):
        CandidateSpec(model_key="lr", priority=1, reason="", evidence_refs=[_ref()], risks=[])

def test_candidate_valid_minimal():
    spec = CandidateSpec(model_key="lr", priority=1, reason="ok", evidence_refs=[_ref()])
    assert spec.risks == []  # default empty list

def test_rejected_requires_non_empty_evidence_refs():
    with pytest.raises(ValidationError):
        RejectedModelSpec(model_key="lr", reason="too complex", evidence_refs=[])

def test_rejected_accepts_optional_reconsider_if():
    spec = RejectedModelSpec(
        model_key="lr", reason="too complex",
        evidence_refs=[_ref()],
        reconsider_if="more data becomes available",
    )
    assert spec.reconsider_if == "more data becomes available"
