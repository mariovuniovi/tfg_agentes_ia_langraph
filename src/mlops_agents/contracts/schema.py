"""Canonical target-schema contract, validated on dataset upload.

Threaded through the pipeline as ``AgentState.schema_json`` (serialised form).
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


class ColumnSchema(BaseModel):
    """A single canonical column. ``extra="allow"`` keeps optional metadata
    (nullable, unique, mapping hints) supplied in the uploaded schema."""

    model_config = ConfigDict(extra="allow")

    name: str
    dtype: str


class SchemaContract(BaseModel):
    """Full target schema: problem type, target, columns, and forecasting keys."""

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
