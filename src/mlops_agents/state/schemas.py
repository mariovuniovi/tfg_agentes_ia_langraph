"""Pydantic schemas for structured LLM outputs and tool I/O."""

from typing import Literal
from pydantic import BaseModel, Field


class RouterOutput(BaseModel):
    """Structured output for the supervisor routing decision.

    The `reasoning` field makes every routing decision auditable —
    useful for thesis analysis and debugging agent behavior.
    """

    next: Literal[
        "data_validator",
        "trainer",
        "evaluator",
        "deployer",
        "FINISH",
    ] = Field(description="The next agent to delegate to, or FINISH to end the pipeline.")
    reasoning: str = Field(description="One sentence explaining why this agent was chosen.")


class ValidationResult(BaseModel):
    """Output schema for the data validation tool."""

    passed: bool
    issues: list[str] = Field(default_factory=list)
    row_count: int
    feature_count: int
    missing_pct: float
    drift_detected: bool
    drift_score: float | None = None
    summary: str


class TrainingResult(BaseModel):
    """Output schema for the model training tool."""

    run_id: str
    model_path: str
    model_type: str
    hyperparameters: dict
    train_accuracy: float
    val_accuracy: float
    summary: str


class EvaluationResult(BaseModel):
    """Output schema for the model evaluation tool."""

    run_id: str
    accuracy: float
    f1_score: float
    auc_roc: float
    precision: float
    recall: float
    beats_baseline: bool
    improvement_pct: float
    recommendation: Literal["promote", "reject", "retrain"]
    summary: str
