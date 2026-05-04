"""Unit tests for SchemaContract Pydantic model."""

import pytest
from pydantic import ValidationError

from mlops_agents.state.schemas import SchemaContract


def _classification_schema(**overrides):
    base = {
        "problem_type": "classification",
        "target_column": "label",
        "columns": [{"name": "feature_a", "dtype": "float"}, {"name": "label", "dtype": "str"}],
    }
    base.update(overrides)
    return base


def _forecasting_schema(**overrides):
    base = {
        "problem_type": "forecasting",
        "target_column": "sales",
        "datetime_column": "date",
        "series_id_columns": ["store_id"],
        "forecast_horizon": 30,
        "frequency": "D",
        "columns": [
            {"name": "date", "dtype": "datetime"},
            {"name": "store_id", "dtype": "str"},
            {"name": "sales", "dtype": "float"},
        ],
    }
    base.update(overrides)
    return base


def test_valid_classification_schema_passes():
    contract = SchemaContract.model_validate(_classification_schema())
    assert contract.problem_type == "classification"
    assert contract.target_column == "label"


def test_valid_regression_schema_passes():
    schema = {
        "problem_type": "regression",
        "target_column": "price",
        "columns": [{"name": "size", "dtype": "float"}, {"name": "price", "dtype": "float"}],
    }
    contract = SchemaContract.model_validate(schema)
    assert contract.problem_type == "regression"


def test_valid_forecasting_schema_passes():
    contract = SchemaContract.model_validate(_forecasting_schema())
    assert contract.problem_type == "forecasting"
    assert contract.forecast_horizon == 30


def test_extra_column_fields_allowed():
    schema = _classification_schema()
    schema["columns"][0]["nullable"] = False
    schema["columns"][0]["description"] = "A feature"
    schema["columns"][0]["mapping_hint"] = "some hint"
    # Must not raise
    SchemaContract.model_validate(schema)


def test_extra_top_level_fields_allowed():
    schema = _classification_schema()
    schema["name"] = "my_dataset"
    schema["description"] = "for testing"
    SchemaContract.model_validate(schema)


def test_missing_problem_type_raises():
    schema = _classification_schema()
    del schema["problem_type"]
    with pytest.raises(ValidationError):
        SchemaContract.model_validate(schema)


def test_invalid_problem_type_raises():
    with pytest.raises(ValidationError):
        SchemaContract.model_validate(_classification_schema(problem_type="clustering"))


def test_target_column_not_in_columns_raises():
    with pytest.raises(ValidationError, match="target_column"):
        SchemaContract.model_validate(_classification_schema(target_column="nonexistent"))


def test_missing_target_column_raises():
    schema = _classification_schema()
    del schema["target_column"]
    with pytest.raises(ValidationError):
        SchemaContract.model_validate(schema)


def test_forecasting_missing_datetime_column_raises():
    schema = _forecasting_schema()
    del schema["datetime_column"]
    with pytest.raises(ValidationError, match="datetime_column"):
        SchemaContract.model_validate(schema)


def test_forecasting_datetime_column_not_in_columns_raises():
    with pytest.raises(ValidationError, match="datetime_column"):
        SchemaContract.model_validate(_forecasting_schema(datetime_column="nonexistent"))


def test_forecasting_zero_horizon_raises():
    with pytest.raises(ValidationError, match="forecast_horizon"):
        SchemaContract.model_validate(_forecasting_schema(forecast_horizon=0))


def test_forecasting_negative_horizon_raises():
    with pytest.raises(ValidationError, match="forecast_horizon"):
        SchemaContract.model_validate(_forecasting_schema(forecast_horizon=-1))


def test_forecasting_missing_frequency_raises():
    schema = _forecasting_schema()
    del schema["frequency"]
    with pytest.raises(ValidationError, match="frequency"):
        SchemaContract.model_validate(schema)


def test_forecasting_series_id_not_in_columns_raises():
    with pytest.raises(ValidationError, match="series_id_columns"):
        SchemaContract.model_validate(_forecasting_schema(series_id_columns=["missing_col"]))


def test_forecasting_empty_series_id_columns_passes():
    # Single-series forecasting: series_id_columns may be []
    schema = _forecasting_schema(series_id_columns=[])
    SchemaContract.model_validate(schema)
