"""Pydantic schemas for structured LLM outputs and tool I/O."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


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


class ColumnSchema(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    dtype: str


class SchemaContract(BaseModel):
    model_config = ConfigDict(extra="allow")

    problem_type: Literal["classification", "regression", "forecasting"]
    target_column: str
    columns: list[ColumnSchema]
    datetime_column: str | None = None
    series_id_columns: list[str] = []
    forecast_horizon: int | None = None
    frequency: str | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> "SchemaContract":
        column_names = {c.name for c in self.columns}
        if self.target_column not in column_names:
            raise ValueError(
                f"'target_column' '{self.target_column}' not found in columns."
            )
        if self.problem_type == "forecasting":
            if not self.datetime_column:
                raise ValueError("'datetime_column' required for forecasting.")
            if self.datetime_column not in column_names:
                raise ValueError(
                    f"'datetime_column' '{self.datetime_column}' not found in columns."
                )
            if self.forecast_horizon is None or self.forecast_horizon <= 0:
                raise ValueError("'forecast_horizon' must be a positive integer.")
            if not self.frequency:
                raise ValueError("'frequency' required for forecasting.")
            for col in self.series_id_columns:
                if col not in column_names:
                    raise ValueError(
                        f"'series_id_columns' entry '{col}' not found in columns."
                    )
        return self
