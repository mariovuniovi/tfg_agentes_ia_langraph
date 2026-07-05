"""Regression: CandidateSpec/RejectedModelSpec must coerce dict evidence_refs into
EvidenceReference instances. Previously typed as `list[Any]`, dicts stayed as dicts
and downstream validation crashed with 'dict' object has no attribute 'source'."""
from mlops_agents.contracts.evidence import EvidenceReference
from mlops_agents.contracts.training import CandidateSpec, RejectedModelSpec


def test_candidate_spec_coerces_dict_evidence_refs():
    spec = CandidateSpec(
        model_key="ets",
        priority=1,
        reason="ok",
        evidence_refs=[{"source": "registry", "source_id": "ets"}],
    )
    assert isinstance(spec.evidence_refs[0], EvidenceReference)
    assert spec.evidence_refs[0].source == "registry"
    assert spec.evidence_refs[0].source_id == "ets"


def test_rejected_model_spec_coerces_dict_evidence_refs():
    spec = RejectedModelSpec(
        model_key="naive",
        reason="too simple",
        evidence_refs=[{"source": "rule", "source_id": "rule_avoid_naive"}],
    )
    assert isinstance(spec.evidence_refs[0], EvidenceReference)
    assert spec.evidence_refs[0].source == "rule"
