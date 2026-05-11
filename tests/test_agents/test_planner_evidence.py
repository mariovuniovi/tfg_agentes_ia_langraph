import pytest
from mlops_agents.agents.planner import (
    PlannerError,
    _check_evidence_references,
    _check_plan_exhaustiveness,
)
from mlops_agents.contracts.planner import (
    EvidenceReference,
    PlannerContext,
    ExperienceSummary,
)
from mlops_agents.contracts.training import (
    TrainingPlan, TrainingPlanCandidate, RejectedModel
)


def _minimal_ctx() -> PlannerContext:
    return PlannerContext(
        current_dataset_profile={},
        task_metadata={},
        available_models=["ridge", "lightgbm_regressor"],
        similar_experiences=[
            ExperienceSummary(
                experience_id="task_001",
                similarity_score=0.8,
                dataset_summary="medium regression",
                models_trained=["ridge", "lightgbm_regressor"],
                best_model="lightgbm_regressor",
                validation_score=0.12,
            )
        ],
        matched_rules=[{"rule_id": "rule_001", "summary": "prefer boosting"}],
    )


def test_valid_experience_reference_passes():
    ctx = _minimal_ctx()
    refs = [EvidenceReference(source="experience", source_id="task_001", summary="used")]
    _check_evidence_references(refs, ctx)  # must not raise


def test_valid_rule_reference_passes():
    ctx = _minimal_ctx()
    refs = [EvidenceReference(source="rule", source_id="rule_001", summary="applied")]
    _check_evidence_references(refs, ctx)


def test_valid_registry_reference_passes():
    ctx = _minimal_ctx()
    refs = [EvidenceReference(source="registry", source_id="ridge", summary="baseline")]
    _check_evidence_references(refs, ctx)


def test_dataset_profile_null_source_id_passes():
    ctx = _minimal_ctx()
    refs = [EvidenceReference(source="dataset_profile", source_id=None, summary="medium")]
    _check_evidence_references(refs, ctx)


def test_task_metadata_null_source_id_passes():
    ctx = _minimal_ctx()
    refs = [EvidenceReference(source="task_metadata", source_id=None, summary="forecasting")]
    _check_evidence_references(refs, ctx)


def test_dataset_profile_non_null_source_id_raises():
    ctx = _minimal_ctx()
    refs = [EvidenceReference(source="dataset_profile", source_id="something", summary="x")]
    with pytest.raises(PlannerError, match="source_id=None"):
        _check_evidence_references(refs, ctx)


def test_task_metadata_non_null_source_id_raises():
    ctx = _minimal_ctx()
    refs = [EvidenceReference(source="task_metadata", source_id="something", summary="x")]
    with pytest.raises(PlannerError, match="source_id=None"):
        _check_evidence_references(refs, ctx)


def test_experience_unknown_source_id_raises():
    ctx = _minimal_ctx()
    refs = [EvidenceReference(source="experience", source_id="fake_999", summary="x")]
    with pytest.raises(PlannerError, match="fake_999"):
        _check_evidence_references(refs, ctx)


def test_rule_unknown_source_id_raises():
    ctx = _minimal_ctx()
    refs = [EvidenceReference(source="rule", source_id="fake_rule", summary="x")]
    with pytest.raises(PlannerError, match="fake_rule"):
        _check_evidence_references(refs, ctx)


def test_registry_unknown_source_id_raises():
    ctx = _minimal_ctx()
    refs = [EvidenceReference(source="registry", source_id="fake_model", summary="x")]
    with pytest.raises(PlannerError, match="fake_model"):
        _check_evidence_references(refs, ctx)


def test_exhaustiveness_passes_when_all_accounted():
    plan = TrainingPlan(
        problem_type="regression",
        candidates=[TrainingPlanCandidate(priority=1, model_key="ridge", reason="baseline")],
        models_not_recommended=[RejectedModel(model_key="lightgbm_regressor", reason="too slow")],
    )
    _check_plan_exhaustiveness(plan, ["ridge", "lightgbm_regressor"])  # must not raise


def test_exhaustiveness_raises_when_model_missing():
    plan = TrainingPlan(
        problem_type="regression",
        candidates=[TrainingPlanCandidate(priority=1, model_key="ridge", reason="baseline")],
        models_not_recommended=[],
    )
    with pytest.raises(PlannerError, match="lightgbm_regressor"):
        _check_plan_exhaustiveness(plan, ["ridge", "lightgbm_regressor"])
