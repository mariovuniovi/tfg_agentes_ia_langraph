"""Tests for planner contracts (SP5 Model Planning Agent)."""

import pytest
from pydantic import ValidationError

from mlops_agents.contracts.planner import (
    CandidateResultCompact,
    EvidenceReference,
    ExperienceSummary,
    PlannerContext,
    PlannerOutput,
)
from mlops_agents.contracts.training import (
    TrainingPlan,
    TrainingPlanCandidate,
)


def _minimal_plan(problem_type: str = "regression") -> TrainingPlan:
    """Helper: create a minimal valid TrainingPlan using get_models_for()."""
    from mlops_agents.models.loader import get_models_for

    models = get_models_for(problem_type)
    candidates = [
        TrainingPlanCandidate(priority=i + 1, model_key=m.model_key, reason="test")
        for i, m in enumerate(models)
    ]
    return TrainingPlan(problem_type=problem_type, candidates=candidates)


class TestEvidenceReference:
    """EvidenceReference validation tests."""

    def test_evidence_reference_valid_experience(self):
        """Experience source with source_id."""
        ref = EvidenceReference(
            source="experience", source_id="task_001", summary="used LightGBM"
        )
        assert ref.source == "experience"
        assert ref.source_id == "task_001"
        assert ref.summary == "used LightGBM"

    def test_evidence_reference_dataset_profile_null_source_id(self):
        """dataset_profile source allows null source_id."""
        ref = EvidenceReference(
            source="dataset_profile", source_id=None, summary="medium dataset"
        )
        assert ref.source == "dataset_profile"
        assert ref.source_id is None
        assert ref.summary == "medium dataset"

    def test_evidence_reference_task_metadata_null_source_id(self):
        """task_metadata source allows null source_id."""
        ref = EvidenceReference(
            source="task_metadata", source_id=None, summary="forecasting task"
        )
        assert ref.source == "task_metadata"
        assert ref.source_id is None

    def test_evidence_reference_rule_source(self):
        """Rule source with rule identifier."""
        ref = EvidenceReference(
            source="rule", source_id="rule_005", summary="high variance → use ensemble"
        )
        assert ref.source == "rule"
        assert ref.source_id == "rule_005"

    def test_evidence_reference_registry_source(self):
        """Registry source."""
        ref = EvidenceReference(
            source="registry", source_id=None, summary="cross-problem-type consensus"
        )
        assert ref.source == "registry"


class TestCandidateResultCompact:
    """CandidateResultCompact validation tests."""

    def test_candidate_result_compact_with_metric(self):
        """Candidate with metric value."""
        c = CandidateResultCompact(model_key="lightgbm_regressor", rank=1, metric_value=0.42)
        assert c.model_key == "lightgbm_regressor"
        assert c.rank == 1
        assert c.metric_value == 0.42

    def test_candidate_result_compact_without_metric(self):
        """Candidate without metric value (e.g., planned but not yet trained)."""
        c = CandidateResultCompact(model_key="ridge", rank=2, metric_value=None)
        assert c.metric_value is None


class TestExperienceSummary:
    """ExperienceSummary validation tests."""

    def test_experience_summary_full(self):
        """ExperienceSummary with all fields."""
        es = ExperienceSummary(
            experience_id="task_001",
            similarity_score=0.84,
            dataset_summary="medium regression dataset",
            models_trained=["ridge", "lightgbm_regressor"],
            best_model="lightgbm_regressor",
            validation_score=0.12,
            notes="boosting worked well",
        )
        assert es.experience_id == "task_001"
        assert es.similarity_score == 0.84
        assert es.best_model == "lightgbm_regressor"
        assert es.candidate_results == []
        assert es.notes == "boosting worked well"

    def test_experience_summary_with_candidate_results(self):
        """ExperienceSummary with candidate results."""
        candidates = [
            CandidateResultCompact(model_key="ridge", rank=1, metric_value=0.42),
            CandidateResultCompact(model_key="linear_svr", rank=2, metric_value=0.38),
        ]
        es = ExperienceSummary(
            experience_id="task_002",
            similarity_score=0.75,
            dataset_summary="small dataset",
            models_trained=["ridge", "linear_svr"],
            best_model="ridge",
            validation_score=0.42,
            candidate_results=candidates,
        )
        assert len(es.candidate_results) == 2
        assert es.candidate_results[0].rank == 1

    def test_experience_summary_empty_notes(self):
        """ExperienceSummary with empty notes (default)."""
        es = ExperienceSummary(
            experience_id="task_003",
            similarity_score=0.6,
            dataset_summary="large dataset",
            models_trained=["xgboost_regressor"],
            best_model="xgboost_regressor",
            validation_score=0.55,
        )
        assert es.notes == ""


class TestPlannerContext:
    """PlannerContext validation tests."""

    def test_planner_context_empty_experiences(self):
        """PlannerContext with empty similar_experiences."""
        ctx = PlannerContext(
            current_dataset_profile={"problem_type": "regression", "n_rows": "medium"},
            task_metadata={"target_column": "y"},
            available_models=["ridge", "lightgbm_regressor"],
            similar_experiences=[],
            matched_rules=[],
        )
        assert ctx.similar_experiences == []
        assert ctx.matched_rules == []
        assert "problem_type" in ctx.current_dataset_profile

    def test_planner_context_with_experiences(self):
        """PlannerContext with similar experiences."""
        exp = ExperienceSummary(
            experience_id="task_001",
            similarity_score=0.9,
            dataset_summary="similar dataset",
            models_trained=["lightgbm_regressor"],
            best_model="lightgbm_regressor",
            validation_score=0.45,
        )
        ctx = PlannerContext(
            current_dataset_profile={"problem_type": "regression"},
            task_metadata={},
            available_models=["lightgbm_regressor"],
            similar_experiences=[exp],
            matched_rules=[{"rule_id": "rule_001", "action": "prefer_ensemble"}],
        )
        assert len(ctx.similar_experiences) == 1
        assert ctx.similar_experiences[0].experience_id == "task_001"
        assert len(ctx.matched_rules) == 1


class TestPlannerOutput:
    """PlannerOutput validation tests."""

    def test_planner_output_requires_plan(self):
        """PlannerOutput.plan is required."""
        with pytest.raises(ValidationError) as exc_info:
            PlannerOutput(planning_analysis="ok")  # missing plan
        assert "plan" in str(exc_info.value).lower()

    def test_planner_output_valid_minimal(self):
        """PlannerOutput with minimal required fields."""
        plan = _minimal_plan("regression")
        out = PlannerOutput(planning_analysis="analysis text", plan=plan)
        assert out.planning_analysis == "analysis text"
        assert out.plan.problem_type == "regression"
        assert out.evidence_used == []
        assert out.risks_or_warnings == []

    def test_planner_output_with_evidence_and_warnings(self):
        """PlannerOutput with evidence and warnings."""
        plan = _minimal_plan("classification")
        evidence = [
            EvidenceReference(
                source="dataset_profile",
                source_id=None,
                summary="imbalanced classes detected",
            ),
            EvidenceReference(
                source="experience",
                source_id="task_042",
                summary="similar dataset preferred ensemble",
            ),
        ]
        warnings = [
            "limited training budget may affect hyperparameter optimization",
            "small sample size — consider stratified cross-validation",
        ]
        out = PlannerOutput(
            planning_analysis="detailed reasoning",
            plan=plan,
            evidence_used=evidence,
            risks_or_warnings=warnings,
        )
        assert len(out.evidence_used) == 2
        assert len(out.risks_or_warnings) == 2
        assert out.evidence_used[0].source == "dataset_profile"

    def test_planner_output_plan_validation_propagates(self):
        """Validation errors in TrainingPlan propagate to PlannerOutput."""
        # Create invalid plan: duplicate priorities
        invalid_plan_dict = {
            "problem_type": "regression",
            "candidates": [
                {"priority": 1, "model_key": "ridge", "reason": "test"},
                {"priority": 1, "model_key": "lightgbm_regressor", "reason": "test"},
            ],
        }
        with pytest.raises(ValidationError):
            PlannerOutput(planning_analysis="bad", plan=invalid_plan_dict)
