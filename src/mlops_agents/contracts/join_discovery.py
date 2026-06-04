"""Pydantic contracts for agentic join discovery."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ColumnProfile(BaseModel):
    column_name: str
    dtype: str
    non_null_count: int
    null_rate: float
    unique_count: int
    unique_ratio: float
    min_value: str | None = None
    max_value: str | None = None


class RawDatasetProfile(BaseModel):
    dataset_name: str
    path: str
    n_rows: int
    n_columns: int
    columns: list[ColumnProfile]
    head_rows: list[dict[str, Any]] = Field(default_factory=list)  # head(profile_nrows) for agent inspection


class BaseDatasetSelection(BaseModel):
    dataset_name: str
    confidence: Literal["high", "medium", "low"]
    covered_target_columns: list[str] = Field(default_factory=list)
    missing_target_columns: list[str] = Field(default_factory=list)
    reason: str = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)


class JoinCandidateEvaluation(BaseModel):
    candidate_id: str
    left_dataset: str
    left_column: str
    right_dataset: str
    right_column: str
    left_distinct: int
    right_distinct: int
    intersection_count: int
    left_coverage: float
    right_coverage: float
    jaccard: float
    containment: float
    left_unique_ratio: float
    right_unique_ratio: float
    inferred_relationship: Literal[
        "one_to_one", "one_to_many", "many_to_one", "many_to_many", "unknown"
    ]
    estimated_inner_rows: int
    estimated_left_rows: int
    row_multiplier_left: float
    join_explosion_risk: Literal["low", "medium", "high"]
    warnings: list[str] = Field(default_factory=list)


class SelectedJoin(BaseModel):
    step_id: int
    candidate_id: str
    left_dataset: str
    left_column: str
    right_dataset: str
    right_column: str
    join_type: Literal["left"] = "left"
    columns_added: list[str] = Field(default_factory=list)
    evaluation: JoinCandidateEvaluation
    confidence_after_evaluation: Literal["high", "medium", "low"]
    reason: str = Field(min_length=1)
    warnings: list[str] = Field(default_factory=list)


class RejectedJoinCandidate(BaseModel):
    candidate_id: str
    left_dataset: str
    left_column: str
    right_dataset: str
    right_column: str
    reason: str = Field(min_length=1)
    evaluation: JoinCandidateEvaluation | None = None


class JoinPlan(BaseModel):
    mode: Literal["explicit", "inferred", "hybrid"] = "inferred"
    base_dataset: BaseDatasetSelection
    selected_joins: list[SelectedJoin] = Field(default_factory=list)
    rejected_candidates: list[RejectedJoinCandidate] = Field(default_factory=list)
    unresolved_ambiguities: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
