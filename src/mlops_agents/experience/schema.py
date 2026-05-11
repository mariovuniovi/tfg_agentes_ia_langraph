"""Pydantic schemas for experience records and retrieval views."""
from __future__ import annotations
from typing import Any, Literal
from pydantic import BaseModel, Field


class CandidateResult(BaseModel):
    model_key: str
    status: Literal["successful", "failed"]
    best_params: dict[str, Any] | None = None
    best_score: float | None = None
    best_score_std: float | None = None
    n_trials_used: int | None = None
    duration_s: float | None = None
    complexity_rank: int | None = None
    mlflow_run_id: str | None = None
    error_type: str | None = None
    error_message: str | None = None


class SelectedSolution(BaseModel):
    model_key: str
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    validation_strategy: str | None = None
    main_metric: str | None = None
    validation_score: float | None = None
    validation_std: float | None = None
    complexity_rank: int | None = None


class ExperienceRecord(BaseModel):
    task_id: str
    problem_type: str
    dataset_name: str | None = None
    dataset_profile: dict[str, Any]
    training_plan_input: dict[str, Any] = Field(default_factory=dict)
    split_artifacts: dict[str, str] = Field(default_factory=dict)
    mlflow: dict[str, str] = Field(default_factory=dict)
    metric_to_optimize: str | None = None
    metric_direction: str | None = None
    candidate_selection_policy: dict[str, Any] = Field(default_factory=dict)
    models_tested: list[CandidateResult] = Field(default_factory=list)
    selected_solution: SelectedSolution | None = None
    experience_summary: str | None = None
    validation_strategy: dict | None = None
    exog_availability: dict | None = None
    exog_strategies: dict | None = None
    per_fold_metrics: list[dict] | None = None
    exog_fit_failures: list[dict] | None = None
    expected_drift: str | None = None
    planner_output: dict | None = None


class CandidateResultView(BaseModel):
    model_key: str
    status: Literal["successful", "failed"]
    best_score: float | None = None
    complexity_rank: int | None = None
    error_type: str | None = None


class SelectedSolutionView(BaseModel):
    model_key: str
    validation_score: float
    validation_std: float | None = None
    complexity_rank: int


class RetrievalView(BaseModel):
    task_id: str
    dataset_name: str | None
    dataset_profile: dict[str, Any]
    models_tested: list[CandidateResultView]
    selected_solution: SelectedSolutionView
    models_not_recommended: list[dict] = Field(default_factory=list)
    experience_summary: str | None
    similarity_score: int
    similarity_ratio: float
    matched_fields: list[str]
