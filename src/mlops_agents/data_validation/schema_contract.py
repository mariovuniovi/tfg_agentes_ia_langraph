"""Deterministic validation of the uploaded schema's ML dataset contract."""

from __future__ import annotations

from typing import Any


def validate_schema_contract(schema_data: dict[str, Any]) -> None:
    """Validate ML dataset contract fields. Raises ValueError on any violation."""
    column_names = {c["name"] for c in schema_data.get("columns", [])}

    problem_type = schema_data.get("problem_type")
    if problem_type not in ("classification", "regression", "forecasting"):
        raise ValueError(
            f"Schema missing or invalid 'problem_type'. Got: {problem_type!r}. "
            "Must be 'classification', 'regression', or 'forecasting'."
        )

    target_column = schema_data.get("target_column")
    if not target_column or target_column not in column_names:
        raise ValueError(
            f"'target_column' must be declared and exist in columns. Got: {target_column!r}."
        )

    if problem_type == "forecasting":
        required = ["datetime_column", "forecast_horizon", "frequency"]
        missing = [f for f in required if schema_data.get(f) is None]
        if missing:
            raise ValueError(f"Forecasting schema missing required fields: {missing}")

        if not isinstance(schema_data["forecast_horizon"], int) or schema_data["forecast_horizon"] <= 0:
            raise ValueError(
                f"'forecast_horizon' must be a positive integer. Got: {schema_data['forecast_horizon']!r}."
            )

        if schema_data["datetime_column"] not in column_names:
            raise ValueError(
                f"'datetime_column' '{schema_data['datetime_column']}' not found in columns."
            )

        for col in schema_data.get("series_id_columns", []):
            if col not in column_names:
                raise ValueError(f"'series_id_columns' entry '{col}' not found in columns.")

    join_policy = schema_data.get("join_policy", {})
    if join_policy:
        valid_modes = {"explicit", "inferred", "hybrid"}
        mode = join_policy.get("mode")
        if mode and mode not in valid_modes:
            raise ValueError(f"join_policy.mode must be one of {valid_modes}, got {mode!r}")
