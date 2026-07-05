"""Cross-cutting Pydantic contracts for the training pipeline.

Used by:
- the deterministic training executor (mlops_agents.training)
- the benchmark runner + experience retrieval
- the planner agent (mlops_agents.planning) — produces TrainingPlan
- graph nodes, which dump these models into AgentState as plain dicts
"""

from __future__ import annotations

from typing import Any, Literal, Self

from pydantic import BaseModel, Field, model_validator

from mlops_agents.contracts.evidence import EvidenceReference

ExogStrategy = Literal["known_future", "naive_carry", "ets", "auto_arima"]
UnknownFutureStrategy = Literal["naive_carry", "ets", "auto_arima"]


class ValidationStrategy(BaseModel):
    type: Literal["single_split", "rolling_window", "expanding_window"] = "single_split"
    n_folds: int = Field(default=1, ge=1)
    horizon: int = Field(ge=1)
    step_size: int | None = None
    window_size: int | None = None


class ExogStrategySettings(BaseModel):
    per_column: dict[str, ExogStrategy] = Field(default_factory=dict)
    default_unknown_future: UnknownFutureStrategy = "naive_carry"


class ForecastingSettings(BaseModel):
    validation_strategy: ValidationStrategy
    exog_strategies: ExogStrategySettings = Field(default_factory=ExogStrategySettings)


class SearchParamOverride(BaseModel):
    """Explicit override entry — no JSON tuple/list ambiguity.

    Provide exactly one of:
    - {low, high}: continuous narrowing for int/float registry params.
    - {choices: [...]}: discrete narrowing for any param type.
    """

    low: int | float | None = None
    high: int | float | None = None
    choices: list[Any] | None = None

    @model_validator(mode="after")
    def either_range_or_choices(self) -> Self:
        has_range = self.low is not None and self.high is not None
        has_choices = self.choices is not None
        if has_range == has_choices:
            raise ValueError("Provide exactly one of {low, high} or {choices}.")
        if has_range and self.low > self.high:  # type: ignore[operator]  # has_range guarantees low/high are not None
            raise ValueError(f"low ({self.low}) must be <= high ({self.high}).")
        return self


class CandidateSpec(BaseModel):
    """A model candidate in a training plan.

    priority must be >= 1 and unique across candidates (enforced by TrainingPlan).
    reason/evidence_refs default to "" / [] so non-planner constructors (default
    plans, tests) stay valid; the planner's stricter requirements are enforced by
    the deterministic validators in mlops_agents.planning.validation.
    """

    priority: int = Field(ge=1)
    model_key: str
    search_space_override: dict[str, SearchParamOverride] | None = None
    reason: str = ""
    evidence_refs: list[EvidenceReference] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


# Backward-compat alias — existing code using TrainingPlanCandidate keeps working
TrainingPlanCandidate = CandidateSpec


class RejectedModelSpec(BaseModel):
    """A model excluded from the training plan.

    evidence_refs defaults to [] so non-planner constructors stay valid; reconsider_if
    is optional. The planner's stricter requirements are enforced by the deterministic
    validators in mlops_agents.planning.validation.
    """

    model_key: str
    reason: str
    evidence_refs: list[EvidenceReference] = Field(default_factory=list)
    reconsider_if: str | None = None


# Backward-compat alias
RejectedModel = RejectedModelSpec


class PlannerTrainingPlan(BaseModel):
    """The planner agent's *decision surface* — only the fields the LLM controls.

    Deliberately omits ``forecasting_settings``: the validation + exogenous-extension
    policy is resolved deterministically by code (``planner_node``) and the LLM must
    not emit it. ``planner_node`` builds the executable :class:`TrainingPlan` from this
    decision plus the resolved settings. This keeps the planner's contract honest —
    it can no longer return a field that is always overwritten and ignored.
    """

    problem_type: Literal["classification", "regression", "forecasting"]
    metric_to_optimize: str | None = None
    candidates: list[CandidateSpec]
    models_not_recommended: list[RejectedModelSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def priorities_unique(self) -> Self:
        priorities = [c.priority for c in self.candidates]
        if len(priorities) != len(set(priorities)):
            raise ValueError(f"Candidate priorities must be unique. Got: {priorities}")
        return self

    @model_validator(mode="after")
    def _check_plan_integrity(self) -> Self:
        cand = {c.model_key for c in self.candidates}
        rej = {r.model_key for r in self.models_not_recommended}

        overlap = cand & rej
        if overlap:
            raise ValueError(
                f"models appear in both candidates and models_not_recommended: {sorted(overlap)}"
            )

        for r in self.models_not_recommended:
            if not r.reason or not r.reason.strip():
                raise ValueError(
                    f"models_not_recommended[{r.model_key}].reason is empty"
                )

        from mlops_agents.models.loader import get_models_for

        valid_keys = {m.model_key for m in get_models_for(self.problem_type)}
        invalid = (cand | rej) - valid_keys
        if invalid:
            raise ValueError(
                f"unknown or wrong-problem-type model_keys: {sorted(invalid)}"
            )

        return self


class TrainingPlan(PlannerTrainingPlan):
    """The executable experiment contract handed to the deterministic executor.

    Extends the planner's decision (:class:`PlannerTrainingPlan`) with
    ``forecasting_settings`` — the code-resolved validation + exogenous policy.
    Only forecasting plans populate it; classification/regression leave it ``None``.
    """

    forecasting_settings: ForecastingSettings | None = None


class TrainingResult(BaseModel):
    """Returned by run_training_plan(...). Embedded in graph state via agent_state."""
    champion_candidate: dict[str, Any]
    champion_model_path: str
    train_pool_path: str
    test_path: str
    split_metadata_path: str
    mlflow_parent_run_id: str
    experience_record_path: str
    champion_metrics: dict[str, float]
    forecast_chart_png: str | None = None  # base64 PNG; only set for forecasting runs
    selection_score: float | None = None   # validation score the champion was selected on (metric_to_optimize units)
