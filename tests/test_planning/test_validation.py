"""Tests for planning/validation.py."""
import pytest

from mlops_agents.contracts.planner import (
    CandidateSpec,
    DecisionBasis,
    EvidenceReference,
    PlannerOutput,
    PlannerValidationContext,
    RejectedModelSpec,
)
from mlops_agents.contracts.training import (
    ExogStrategySettings,
    ForecastingSettings,
    TrainingPlan,
    ValidationStrategy,
)
from mlops_agents.planning.trace import ToolTrace
from mlops_agents.planning.validation import (
    PlannerValidationError,
    _check_evidence_references_hybrid,
    _check_plan_exhaustiveness,
    _check_plan_integrity,
    _collect_all_refs,
    validate_forecasting_settings,
)

# ---------------------------------------------------------------------------
# Real model keys from the registry (classification problem_type).
# Using these avoids the TrainingPlan._check_plan_integrity validator rejecting
# unknown keys.
# ---------------------------------------------------------------------------
KEY_A = "logistic_regression"
KEY_B = "random_forest_classifier"


def _registry_ref(key: str) -> EvidenceReference:
    return EvidenceReference(source="registry", source_id=key)


def _profile_ref() -> EvidenceReference:
    return EvidenceReference(source="dataset_profile", source_id=None)


def _candidate(key: str, priority: int = 1) -> CandidateSpec:
    return CandidateSpec(model_key=key, priority=priority, reason="ok",
                         evidence_refs=[_registry_ref(key)], risks=[])


def _rejected(key: str) -> RejectedModelSpec:
    return RejectedModelSpec(model_key=key, reason="too complex",
                             evidence_refs=[_registry_ref(key)])


def _minimal_output(candidates=None, rejected=None) -> PlannerOutput:
    candidates = candidates or [_candidate(KEY_A)]
    rejected = rejected or []
    return PlannerOutput(
        planning_analysis="ok",
        decision_basis=DecisionBasis(
            primary_evidence=[_profile_ref()],
            secondary_evidence=[],
            final_strategy="strat",
        ),
        evidence_used=[],
        evidence_conflicts=[],
        risks_or_warnings=[],
        plan=TrainingPlan(
            problem_type="classification",
            candidates=candidates,
            models_not_recommended=rejected,
        ),
    )


def _minimal_ctx(available=None) -> PlannerValidationContext:
    available = available or [KEY_A, KEY_B]
    return PlannerValidationContext(
        problem_type="classification",
        task_metadata={},
        available_model_keys=available,
        available_model_specs=[],
        similar_experiences=[],
        matched_rules=[],
        rules_by_id={},
    )


def _trace_with_required() -> ToolTrace:
    t = ToolTrace()
    t.called_tools = ["list_available_models", "retrieve_similar_experiences", "retrieve_ml_knowledge"]
    t.tool_call_count = 3
    return t


# ---------------------------------------------------------------------------
# _check_plan_integrity tests
# ---------------------------------------------------------------------------

def test_integrity_missing_required_tool_raises():
    out = _minimal_output()
    ctx = _minimal_ctx([KEY_A])
    trace = ToolTrace()
    trace.called_tools = ["list_available_models"]  # missing 2 required
    with pytest.raises(PlannerValidationError, match="required tools"):
        _check_plan_integrity(out, trace, ctx)


def test_integrity_duplicate_priorities_raises():
    # TrainingPlan's own model_validator fires first and rejects non-unique priorities.
    with pytest.raises(Exception):
        _minimal_output(candidates=[_candidate(KEY_A, priority=1), _candidate(KEY_B, priority=1)])


def test_integrity_duplicate_priorities_validation_layer_raises():
    # Bypass TrainingPlan construction; inject duplicate priorities post-hoc.
    out = _minimal_output(candidates=[_candidate(KEY_A, priority=1)])
    # Inject a second candidate with same priority by mutating the plan object directly.
    out.plan.candidates.append(_candidate(KEY_B, priority=1))
    with pytest.raises(PlannerValidationError, match="unique"):
        _check_plan_integrity(out, _trace_with_required(), _minimal_ctx([KEY_A, KEY_B]))


def test_integrity_overlap_candidate_rejected_raises():
    # TrainingPlan's own model_validator catches the overlap first.
    with pytest.raises(Exception):
        _minimal_output(candidates=[_candidate(KEY_A)], rejected=[_rejected(KEY_A)])


def test_integrity_overlap_validation_layer_raises():
    # Bypass TrainingPlan by mutating post-construction.
    out = _minimal_output(candidates=[_candidate(KEY_A)])
    out.plan.models_not_recommended.append(_rejected(KEY_A))
    with pytest.raises(PlannerValidationError, match="overlap"):
        _check_plan_integrity(out, _trace_with_required(), _minimal_ctx([KEY_A]))


def test_integrity_missing_registry_self_cite_on_candidate_raises():
    cand = CandidateSpec(model_key=KEY_A, priority=1, reason="ok",
                         evidence_refs=[_profile_ref()], risks=[])
    out = _minimal_output(candidates=[cand])
    with pytest.raises(PlannerValidationError, match="registry self-citation"):
        _check_plan_integrity(out, _trace_with_required(), _minimal_ctx([KEY_A]))


def test_integrity_passes_on_clean_plan():
    out = _minimal_output()
    _check_plan_integrity(out, _trace_with_required(), _minimal_ctx([KEY_A]))  # no raise


def test_integrity_inspect_cap_exceeded_raises(monkeypatch):
    from mlops_agents.config.settings import settings
    monkeypatch.setattr(settings, "planner_max_inspect_calls", 2)
    out = _minimal_output()
    trace = _trace_with_required()
    trace.inspect_model_details_count = 3  # exceeds cap of 2
    with pytest.raises(PlannerValidationError, match="planner_max_inspect_calls"):
        _check_plan_integrity(out, trace, _minimal_ctx([KEY_A]))


def test_integrity_global_tool_call_cap_exceeded_raises(monkeypatch):
    from mlops_agents.config.settings import settings
    monkeypatch.setattr(settings, "planner_max_tool_calls", 5)
    out = _minimal_output()
    trace = _trace_with_required()
    trace.tool_call_count = 10
    with pytest.raises(PlannerValidationError, match="planner_max_tool_calls"):
        _check_plan_integrity(out, trace, _minimal_ctx([KEY_A]))


# ---------------------------------------------------------------------------
# _check_plan_exhaustiveness tests
# ---------------------------------------------------------------------------

def test_exhaustiveness_missing_model_raises():
    out = _minimal_output(candidates=[_candidate(KEY_A)])
    with pytest.raises(PlannerValidationError, match=KEY_B):
        _check_plan_exhaustiveness(out.plan, [KEY_A, KEY_B])


def test_exhaustiveness_all_accounted_passes():
    out = _minimal_output(
        candidates=[_candidate(KEY_A)],
        rejected=[_rejected(KEY_B)],
    )
    _check_plan_exhaustiveness(out.plan, [KEY_A, KEY_B])  # no raise


def test_exhaustiveness_candidate_only_passes():
    out = _minimal_output(candidates=[_candidate(KEY_A)])
    _check_plan_exhaustiveness(out.plan, [KEY_A])  # no raise


# ---------------------------------------------------------------------------
# _check_evidence_references_hybrid tests
# ---------------------------------------------------------------------------

def test_evidence_ref_registry_unknown_key_raises():
    cand = CandidateSpec(
        model_key=KEY_A, priority=1, reason="ok",
        evidence_refs=[_registry_ref(KEY_A), _registry_ref("nonexistent")],
        risks=[],
    )
    out = _minimal_output(candidates=[cand])
    with pytest.raises(PlannerValidationError, match="nonexistent"):
        _check_evidence_references_hybrid(out, _minimal_ctx([KEY_A]), _trace_with_required())


def test_evidence_ref_experience_not_retrieved_raises():
    ref = EvidenceReference(source="experience", source_id="exp_999")
    out = _minimal_output()
    out.evidence_used = [ref]
    trace = _trace_with_required()
    # exp_999 not in trace.retrieved_experience_ids
    with pytest.raises(PlannerValidationError, match="exp_999"):
        _check_evidence_references_hybrid(out, _minimal_ctx([KEY_A]), trace)


def test_evidence_ref_experience_retrieved_passes():
    ref = EvidenceReference(source="experience", source_id="exp_001")
    out = _minimal_output()
    out.evidence_used = [ref]
    trace = _trace_with_required()
    trace.retrieved_experience_ids = ["exp_001"]
    _check_evidence_references_hybrid(out, _minimal_ctx([KEY_A]), trace)  # no raise


def test_evidence_ref_rule_not_retrieved_raises():
    ref = EvidenceReference(source="rule", source_id="rule_99")
    out = _minimal_output()
    out.evidence_used = [ref]
    trace = _trace_with_required()
    with pytest.raises(PlannerValidationError, match="rule_99"):
        _check_evidence_references_hybrid(out, _minimal_ctx([KEY_A]), trace)


def test_evidence_ref_rule_retrieved_passes():
    ref = EvidenceReference(source="rule", source_id="rule_01")
    out = _minimal_output()
    out.evidence_used = [ref]
    trace = _trace_with_required()
    trace.retrieved_rule_ids = ["rule_01"]
    _check_evidence_references_hybrid(out, _minimal_ctx([KEY_A]), trace)  # no raise


def test_evidence_ref_dataset_profile_with_source_id_raises():
    ref = EvidenceReference(source="dataset_profile", source_id="bad_id")
    out = _minimal_output()
    out.evidence_used = [ref]
    with pytest.raises(PlannerValidationError, match="source_id=None"):
        _check_evidence_references_hybrid(out, _minimal_ctx([KEY_A]), _trace_with_required())


# ---------------------------------------------------------------------------
# _collect_all_refs tests
# ---------------------------------------------------------------------------

def test_collect_all_refs_includes_decision_basis_and_candidate_refs():
    out = _minimal_output()
    refs = _collect_all_refs(out)
    sources = {r.source for r in refs}
    assert "dataset_profile" in sources  # from decision_basis.primary_evidence
    assert "registry" in sources         # from candidate.evidence_refs


def test_collect_all_refs_includes_rejected_refs():
    out = _minimal_output(
        candidates=[_candidate(KEY_A)],
        rejected=[_rejected(KEY_B)],
    )
    refs = _collect_all_refs(out)
    registry_ids = {r.source_id for r in refs if r.source == "registry"}
    assert KEY_A in registry_ids
    assert KEY_B in registry_ids


# ---------------------------------------------------------------------------
# validate_forecasting_settings — validates the CODE-resolved forecasting
# settings (moved here from _check_plan_integrity after the planner/executor
# schema split; the LLM no longer emits forecasting_settings).
# ---------------------------------------------------------------------------


def _fs(per_column=None):
    return ForecastingSettings(
        validation_strategy=ValidationStrategy(type="expanding_window", n_folds=5, horizon=8),
        exog_strategies=ExogStrategySettings(per_column=per_column or {}),
    )


def test_validate_forecasting_settings_passes_on_valid():
    validate_forecasting_settings(_fs(per_column={"temp": "ets"}), {})


def test_validate_forecasting_settings_rejects_unallowed_exog_strategy():
    # "known_future" is a valid ExogStrategy literal but not allowed as an
    # *extension* strategy in per_column.
    with pytest.raises(PlannerValidationError, match="invalid exog strategy"):
        validate_forecasting_settings(_fs(per_column={"temp": "known_future"}), {})


def test_validate_forecasting_settings_rejects_known_future_col_in_per_column():
    # Real metadata shape: exogenous_columns with future_availability (not a flat
    # known_future_columns list, which nothing in the codebase populates).
    with pytest.raises(PlannerValidationError, match="known_future column"):
        validate_forecasting_settings(
            _fs(per_column={"temp": "ets"}),
            {"exogenous_columns": [{"name": "temp", "future_availability": "known_future"}]},
        )


def test_validate_forecasting_settings_rejects_bad_validation_strategy():
    # Bypass the Literal type guard to exercise the defense-in-depth branch.
    fs = _fs()
    fs.validation_strategy = ValidationStrategy.model_construct(
        type="bogus", n_folds=5, horizon=8
    )
    with pytest.raises(PlannerValidationError, match="invalid validation_strategy"):
        validate_forecasting_settings(fs, {})


def test_validate_forecasting_settings_handles_none_exogenous_columns():
    # Datasets with no exog carry exogenous_columns=None (present, not absent).
    # Must not raise "'NoneType' object is not iterable" (regression: small_monthly_revenue).
    validate_forecasting_settings(_fs(), {"exogenous_columns": None})
    validate_forecasting_settings(_fs(), {})  # absent key path too
