"""Honesty tests for the planner/executor schema split.

PlannerOutput describes the *agent's decision* (which models, how many trials).
TrainingPlan describes the *executable experiment contract* (decision + the
code-resolved forecasting policy). The planner LLM must not be able to emit
forecasting_settings, because validation/exog policy is fixed deterministically
by planner_node and would always be overwritten.
"""

from mlops_agents.contracts.planner import PlannerOutput
from mlops_agents.contracts.training import (
    CandidateSpec,
    EvidenceReference,
    ForecastingSettings,
    PlannerTrainingPlan,
    TrainingPlan,
    ValidationStrategy,
)


def _candidates():
    return [CandidateSpec(model_key="ets", priority=1, reason="ok",
                          evidence_refs=[EvidenceReference(source="registry", source_id="ets")])]


# --- the LLM-facing schema must not expose forecasting_settings ---

def test_planner_training_plan_has_no_forecasting_settings_field():
    assert "forecasting_settings" not in PlannerTrainingPlan.model_fields


def test_planner_output_json_schema_excludes_forecasting_settings():
    """The schema the LLM is bound to (response_format=PlannerOutput) must not
    mention forecasting_settings anywhere — the model can't be prompted to emit it."""
    schema = PlannerOutput.model_json_schema()
    plan_schema = schema["$defs"]["PlannerTrainingPlan"]
    assert "forecasting_settings" not in plan_schema.get("properties", {})


def test_planner_training_plan_ignores_stray_forecasting_settings():
    """If a model tries to smuggle forecasting_settings in, Pydantic drops it
    (extra='ignore' default) — it never reaches the decision object."""
    plan = PlannerTrainingPlan(
        problem_type="forecasting",
        candidates=_candidates(),
        forecasting_settings={"validation_strategy": {"type": "expanding_window",
                                                      "n_folds": 5, "horizon": 8}},
    )
    assert not hasattr(plan, "forecasting_settings")
    assert "forecasting_settings" not in plan.model_dump()


# --- the executable contract keeps forecasting_settings ---

def test_training_plan_still_carries_forecasting_settings():
    plan = TrainingPlan(
        problem_type="forecasting",
        candidates=_candidates(),
        forecasting_settings=ForecastingSettings(
            validation_strategy=ValidationStrategy(type="expanding_window", n_folds=5, horizon=8)
        ),
    )
    assert plan.forecasting_settings is not None
    assert "forecasting_settings" in plan.model_dump()


def test_training_plan_is_a_planner_training_plan():
    """Executable contract = decision + settings, so it shares the integrity
    validators (priorities_unique, plan integrity) via inheritance."""
    assert issubclass(TrainingPlan, PlannerTrainingPlan)


def test_executor_plan_built_from_planner_decision_round_trips():
    """planner_node builds TrainingPlan(**decision, forecasting_settings=fs); the
    executor later reconstructs it via TrainingPlan.model_validate. Verify that path."""
    decision = PlannerTrainingPlan(problem_type="forecasting", candidates=_candidates())
    fs = ForecastingSettings(
        validation_strategy=ValidationStrategy(type="expanding_window", n_folds=5, horizon=8)
    )
    executor_plan = TrainingPlan(**decision.model_dump(), forecasting_settings=fs)
    rebuilt = TrainingPlan.model_validate(executor_plan.model_dump())
    assert rebuilt.forecasting_settings.validation_strategy.n_folds == 5
    assert [c.model_key for c in rebuilt.candidates] == ["ets"]
