"""Cross-cutting Pydantic contracts for the training pipeline.

Used by:
- SP3 deterministic executor
- SP4 benchmark runner + retrieval
- SP5 planner agent (future)
- Graph state (via re-export in agent_state)
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


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
    def either_range_or_choices(self):
        has_range = self.low is not None and self.high is not None
        has_choices = self.choices is not None
        if has_range == has_choices:
            raise ValueError("Provide exactly one of {low, high} or {choices}.")
        if has_range and self.low > self.high:
            raise ValueError(f"low ({self.low}) must be <= high ({self.high}).")
        return self


class TrainingPlanCandidate(BaseModel):
    priority: int
    model_key: str
    initial_hyperparameters: dict[str, Any] = Field(default_factory=dict)
    search_space_override: dict[str, SearchParamOverride] | None = None
    requested_trials: int | None = None
    reason: str = ""


class RejectedModel(BaseModel):
    model_key: str
    reason: str


class TrialBudget(BaseModel):
    total_trials: int = 60
    allocation_strategy: Literal["priority_weighted", "equal"] = "priority_weighted"
    max_trials_per_candidate: int = 30
    min_trials_per_candidate: int = 5


class TrainingPlan(BaseModel):
    problem_type: Literal["classification", "regression", "forecasting"]
    metric_to_optimize: str | None = None
    candidates: list[TrainingPlanCandidate]
    models_not_recommended: list[RejectedModel] = Field(default_factory=list)
    trial_budget: TrialBudget = Field(default_factory=TrialBudget)
    validation_strategy: dict[str, Any] | None = None
    forecasting_settings: dict[str, Any] | None = None

    @model_validator(mode="after")
    def priorities_unique(self):
        priorities = [c.priority for c in self.candidates]
        if len(priorities) != len(set(priorities)):
            raise ValueError(f"Candidate priorities must be unique. Got: {priorities}")
        return self


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
