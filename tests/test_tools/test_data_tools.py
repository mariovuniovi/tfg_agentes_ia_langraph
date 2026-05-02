"""Unit tests for data tools (deterministic — no LLM calls)."""

import json
from pathlib import Path

import pandas as pd
import pytest

from mlops_agents.tools.data_tools import (
    apply_column_mapping,
    check_missing_values,
    load_dataset,
    merge_datasets,
    validate_against_schema,
)


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------

def test_load_dataset_returns_row_count(sample_csv):
    result = json.loads(load_dataset.invoke({"dataset_path": str(sample_csv)}))
    assert result["row_count"] == 5


def test_load_dataset_returns_column_names(sample_csv):
    result = json.loads(load_dataset.invoke({"dataset_path": str(sample_csv)}))
    assert "feature_1" in result["column_names"]
    assert "feature_2" in result["column_names"]
    assert "target" in result["column_names"]


def test_load_dataset_returns_column_count(sample_csv):
    result = json.loads(load_dataset.invoke({"dataset_path": str(sample_csv)}))
    assert result["column_count"] == 3


def test_load_dataset_includes_head_rows(sample_csv):
    result = json.loads(load_dataset.invoke({"dataset_path": str(sample_csv)}))
    assert "head" in result
    assert isinstance(result["head"], list)
    assert len(result["head"]) == 3  # head(3) default


def test_load_dataset_returns_dtypes(sample_csv):
    result = json.loads(load_dataset.invoke({"dataset_path": str(sample_csv)}))
    assert "dtypes" in result
    assert "target" in result["dtypes"]


def test_load_dataset_returns_error_for_missing_file():
    result = json.loads(load_dataset.invoke({"dataset_path": "/nonexistent/path.csv"}))
    assert "error" in result


# ---------------------------------------------------------------------------
# check_missing_values
# ---------------------------------------------------------------------------

def test_check_missing_values_no_missing(sample_csv):
    result = json.loads(check_missing_values.invoke({"dataset_path": str(sample_csv)}))
    assert result["columns_with_missing"] == 0
    assert result["max_missing_pct"] == 0.0


def test_check_missing_values_detects_missing(sample_csv_with_missing):
    result = json.loads(check_missing_values.invoke({"dataset_path": str(sample_csv_with_missing)}))
    assert result["columns_with_missing"] == 1
    assert "feature_2" in result["per_column"]
    assert result["per_column"]["feature_2"]["count"] == 2
    assert result["per_column"]["feature_2"]["pct"] == 40.0


def test_check_missing_values_returns_total_rows(sample_csv):
    result = json.loads(check_missing_values.invoke({"dataset_path": str(sample_csv)}))
    assert result["total_rows"] == 5


def test_check_missing_values_passed_threshold_false_when_high_missing(sample_csv_with_missing):
    """40% missing in feature_2 should fail the threshold (MAX_DRIFT_SCORE * 100 = 10%)."""
    result = json.loads(check_missing_values.invoke({"dataset_path": str(sample_csv_with_missing)}))
    assert result["passed_threshold"] is False


def test_check_missing_values_passed_threshold_true_when_clean(sample_csv):
    result = json.loads(check_missing_values.invoke({"dataset_path": str(sample_csv)}))
    assert result["passed_threshold"] is True


# ---------------------------------------------------------------------------
# validate_against_schema
# ---------------------------------------------------------------------------

def test_validate_against_schema_passes_valid_data(canonical_iris_csv, iris_schema_file):
    from mlops_agents.tools.data_tools import validate_against_schema
    result = json.loads(validate_against_schema.invoke({
        "canonical_path": str(canonical_iris_csv),
        "schema_path": str(iris_schema_file),
    }))
    assert result["passed"] is True
    assert result["violations"] == []


def test_validate_against_schema_detects_nullable_violation(tmp_path, iris_schema_file):
    from mlops_agents.tools.data_tools import validate_against_schema
    df = pd.DataFrame({
        "sepal_length": [5.1, None],
        "sepal_width":  [3.5, 3.0],
        "petal_length": [1.4, 1.4],
        "petal_width":  [0.2, 0.2],
        "sample_id":    [1, 2],
        "target":       ["setosa", "setosa"],
    })
    path = tmp_path / "nullable_violation.csv"
    df.to_csv(path, index=False)
    result = json.loads(validate_against_schema.invoke({
        "canonical_path": str(path),
        "schema_path": str(iris_schema_file),
    }))
    assert result["passed"] is False
    assert any(v["column"] == "sepal_length" and v["rule"] == "nullable" for v in result["violations"])


def test_validate_against_schema_detects_min_violation(tmp_path, iris_schema_file):
    from mlops_agents.tools.data_tools import validate_against_schema
    df = pd.DataFrame({
        "sepal_length": [-1.0, 4.9],
        "sepal_width":  [3.5, 3.0],
        "petal_length": [1.4, 1.4],
        "petal_width":  [0.2, 0.2],
        "sample_id":    [1, 2],
        "target":       ["setosa", "setosa"],
    })
    path = tmp_path / "min_violation.csv"
    df.to_csv(path, index=False)
    result = json.loads(validate_against_schema.invoke({
        "canonical_path": str(path),
        "schema_path": str(iris_schema_file),
    }))
    assert result["passed"] is False
    assert any(v["column"] == "sepal_length" and v["rule"] == "min" for v in result["violations"])


def test_validate_against_schema_detects_allowed_values_violation(tmp_path, iris_schema_file):
    from mlops_agents.tools.data_tools import validate_against_schema
    df = pd.DataFrame({
        "sepal_length": [5.1],
        "sepal_width":  [3.5],
        "petal_length": [1.4],
        "petal_width":  [0.2],
        "sample_id":    [1],
        "target":       ["unknown_species"],
    })
    path = tmp_path / "allowed_violation.csv"
    df.to_csv(path, index=False)
    result = json.loads(validate_against_schema.invoke({
        "canonical_path": str(path),
        "schema_path": str(iris_schema_file),
    }))
    assert result["passed"] is False
    assert any(v["column"] == "target" and v["rule"] == "allowed_values" for v in result["violations"])


def test_validate_against_schema_detects_missing_required_column(tmp_path, iris_schema_file):
    from mlops_agents.tools.data_tools import validate_against_schema
    df = pd.DataFrame({
        "sepal_length": [5.1],
        "sepal_width":  [3.5],
        # petal_length missing
        "petal_width":  [0.2],
        "sample_id":    [1],
        "target":       ["setosa"],
    })
    path = tmp_path / "missing_col.csv"
    df.to_csv(path, index=False)
    result = json.loads(validate_against_schema.invoke({
        "canonical_path": str(path),
        "schema_path": str(iris_schema_file),
    }))
    assert result["passed"] is False
    assert any(v["column"] == "petal_length" and v["rule"] == "required" for v in result["violations"])


# ---------------------------------------------------------------------------
# apply_column_mapping
# ---------------------------------------------------------------------------

def test_apply_column_mapping_renames_columns(tmp_path, sample_csv):
    from mlops_agents.tools.data_tools import apply_column_mapping
    mapping = json.dumps({"feature_1": "sepal_length", "feature_2": "sepal_width", "target": "target"})
    output_path = str(tmp_path / "canonical.csv")
    result = json.loads(apply_column_mapping.invoke({
        "raw_path": str(sample_csv),
        "mapping_json": mapping,
        "output_path": output_path,
    }))
    assert result["success"] is True
    out_df = pd.read_csv(output_path)
    assert "sepal_length" in out_df.columns
    assert "sepal_width" in out_df.columns
    assert "feature_1" not in out_df.columns


def test_apply_column_mapping_drops_unmapped_columns(tmp_path, sample_csv):
    from mlops_agents.tools.data_tools import apply_column_mapping
    mapping = json.dumps({"feature_1": "sepal_length"})
    output_path = str(tmp_path / "canonical.csv")
    result = json.loads(apply_column_mapping.invoke({
        "raw_path": str(sample_csv),
        "mapping_json": mapping,
        "output_path": output_path,
    }))
    assert result["success"] is True
    out_df = pd.read_csv(output_path)
    assert list(out_df.columns) == ["sepal_length"]
    assert "feature_2" not in out_df.columns
    assert "dropped_columns" in result


def test_apply_column_mapping_writes_output_file(tmp_path, sample_csv):
    from mlops_agents.tools.data_tools import apply_column_mapping
    mapping = json.dumps({"feature_1": "sepal_length", "target": "target"})
    output_path = str(tmp_path / "out.csv")
    apply_column_mapping.invoke({
        "raw_path": str(sample_csv),
        "mapping_json": mapping,
        "output_path": output_path,
    })
    assert Path(output_path).exists()


def test_apply_column_mapping_reports_mapped_columns(tmp_path, sample_csv):
    from mlops_agents.tools.data_tools import apply_column_mapping
    mapping = json.dumps({"feature_1": "sepal_length", "target": "target"})
    output_path = str(tmp_path / "out.csv")
    result = json.loads(apply_column_mapping.invoke({
        "raw_path": str(sample_csv),
        "mapping_json": mapping,
        "output_path": output_path,
    }))
    assert "mapped_columns" in result
    assert "sepal_length" in result["mapped_columns"]


# ---------------------------------------------------------------------------
# merge_datasets
# ---------------------------------------------------------------------------

def test_merge_datasets_joins_two_files(tmp_path, measurements_csv, labels_csv):
    from mlops_agents.tools.data_tools import merge_datasets
    join_spec = json.dumps({
        "join_key": "sample_id",
        "files": [
            {"path": str(measurements_csv), "key_column": "Id"},
            {"path": str(labels_csv),       "key_column": "sample_id"},
        ],
    })
    output_path = str(tmp_path / "merged.csv")
    result = json.loads(merge_datasets.invoke({
        "join_spec_json": join_spec,
        "output_path": output_path,
    }))
    assert result["success"] is True
    merged = pd.read_csv(output_path)
    assert "SepalLengthCm" in merged.columns
    assert "species" in merged.columns
    assert result["row_count"] == 3


def test_merge_datasets_writes_output_file(tmp_path, measurements_csv, labels_csv):
    from mlops_agents.tools.data_tools import merge_datasets
    join_spec = json.dumps({
        "join_key": "sample_id",
        "files": [
            {"path": str(measurements_csv), "key_column": "Id"},
            {"path": str(labels_csv),       "key_column": "sample_id"},
        ],
    })
    output_path = str(tmp_path / "merged.csv")
    merge_datasets.invoke({"join_spec_json": join_spec, "output_path": output_path})
    assert Path(output_path).exists()


def test_merge_datasets_returns_error_when_key_missing(tmp_path, measurements_csv, labels_csv):
    from mlops_agents.tools.data_tools import merge_datasets
    join_spec = json.dumps({
        "join_key": "sample_id",
        "files": [
            {"path": str(measurements_csv), "key_column": "nonexistent_key"},
            {"path": str(labels_csv),       "key_column": "sample_id"},
        ],
    })
    output_path = str(tmp_path / "merged.csv")
    result = json.loads(merge_datasets.invoke({
        "join_spec_json": join_spec,
        "output_path": output_path,
    }))
    assert "error" in result
    assert not Path(output_path).exists()


def test_merge_datasets_returns_error_when_join_produces_zero_rows(tmp_path):
    from mlops_agents.tools.data_tools import merge_datasets
    df_a = pd.DataFrame({"id": [1, 2], "val_a": [10, 20]})
    df_b = pd.DataFrame({"id": [3, 4], "val_b": ["x", "y"]})
    path_a = tmp_path / "a.csv"
    path_b = tmp_path / "b.csv"
    df_a.to_csv(path_a, index=False)
    df_b.to_csv(path_b, index=False)
    join_spec = json.dumps({
        "join_key": "id",
        "files": [
            {"path": str(path_a), "key_column": "id"},
            {"path": str(path_b), "key_column": "id"},
        ],
    })
    output_path = str(tmp_path / "merged.csv")
    result = json.loads(merge_datasets.invoke({
        "join_spec_json": join_spec,
        "output_path": output_path,
    }))
    assert "error" in result


# ---------------------------------------------------------------------------
# impute_missing_values
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock
import mlops_agents.tools.data_tools as _dt


def _make_settings(numeric: str = "mean", categorical: str = "mode") -> MagicMock:
    s = MagicMock()
    s.imputation_strategy_numeric = numeric
    s.imputation_strategy_categorical = categorical
    return s


def test_impute_numeric_mean(tmp_path, monkeypatch):
    from mlops_agents.tools.data_tools import impute_missing_values

    df = pd.DataFrame({"sepal_width": [3.0, None, 5.0], "target": ["a", "b", "c"]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    monkeypatch.setattr(_dt, "settings", _make_settings(numeric="mean"))
    result = json.loads(impute_missing_values.invoke({"path": str(path)}))

    assert "sepal_width" in result["imputed_columns"]
    assert result["imputed_columns"]["sepal_width"]["strategy"] == "mean"
    assert result["imputed_columns"]["sepal_width"]["rows_affected"] == 1
    assert abs(result["imputed_columns"]["sepal_width"]["fill_value"] - 4.0) < 0.01
    df_after = pd.read_csv(path)
    assert df_after["sepal_width"].isnull().sum() == 0


def test_impute_numeric_median(tmp_path, monkeypatch):
    from mlops_agents.tools.data_tools import impute_missing_values

    df = pd.DataFrame({"val": [1.0, None, 9.0, None, 5.0]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    monkeypatch.setattr(_dt, "settings", _make_settings(numeric="median"))
    result = json.loads(impute_missing_values.invoke({"path": str(path)}))

    assert result["imputed_columns"]["val"]["strategy"] == "median"
    assert result["imputed_columns"]["val"]["rows_affected"] == 2
    df_after = pd.read_csv(path)
    assert df_after["val"].isnull().sum() == 0


def test_impute_numeric_zero(tmp_path, monkeypatch):
    from mlops_agents.tools.data_tools import impute_missing_values

    df = pd.DataFrame({"val": [1.0, None, 3.0]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    monkeypatch.setattr(_dt, "settings", _make_settings(numeric="zero"))
    result = json.loads(impute_missing_values.invoke({"path": str(path)}))

    assert result["imputed_columns"]["val"]["fill_value"] == 0.0
    df_after = pd.read_csv(path)
    assert df_after["val"].isnull().sum() == 0


def test_impute_categorical_mode(tmp_path, monkeypatch):
    from mlops_agents.tools.data_tools import impute_missing_values

    df = pd.DataFrame({"species": ["setosa", "setosa", None, "virginica"]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    monkeypatch.setattr(_dt, "settings", _make_settings(categorical="mode"))
    result = json.loads(impute_missing_values.invoke({"path": str(path)}))

    assert result["imputed_columns"]["species"]["fill_value"] == "setosa"
    df_after = pd.read_csv(path)
    assert df_after["species"].isnull().sum() == 0


def test_impute_categorical_unknown(tmp_path, monkeypatch):
    from mlops_agents.tools.data_tools import impute_missing_values

    df = pd.DataFrame({"species": ["setosa", None]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    monkeypatch.setattr(_dt, "settings", _make_settings(categorical="unknown"))
    result = json.loads(impute_missing_values.invoke({"path": str(path)}))

    assert result["imputed_columns"]["species"]["fill_value"] == "unknown"


def test_impute_categorical_drop_row(tmp_path, monkeypatch):
    from mlops_agents.tools.data_tools import impute_missing_values

    df = pd.DataFrame({"species": ["setosa", None, "virginica"]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    monkeypatch.setattr(_dt, "settings", _make_settings(categorical="drop_row"))
    result = json.loads(impute_missing_values.invoke({"path": str(path)}))

    assert result["imputed_columns"]["species"]["strategy"] == "drop_row"
    df_after = pd.read_csv(path)
    assert len(df_after) == 2


def test_impute_no_missing_is_noop(tmp_path, monkeypatch):
    from mlops_agents.tools.data_tools import impute_missing_values

    df = pd.DataFrame({"val": [1.0, 2.0, 3.0]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    monkeypatch.setattr(_dt, "settings", _make_settings())
    result = json.loads(impute_missing_values.invoke({"path": str(path)}))

    assert result["imputed_columns"] == {}


def test_impute_returns_output_path(tmp_path, monkeypatch):
    from mlops_agents.tools.data_tools import impute_missing_values

    df = pd.DataFrame({"val": [1.0, None]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)

    monkeypatch.setattr(_dt, "settings", _make_settings())
    result = json.loads(impute_missing_values.invoke({"path": str(path)}))

    assert result["output_path"] == str(path)


def test_impute_file_not_found(tmp_path, monkeypatch):
    from mlops_agents.tools.data_tools import impute_missing_values

    monkeypatch.setattr(_dt, "settings", _make_settings())
    result = json.loads(impute_missing_values.invoke({"path": "/nonexistent/file.csv"}))

    assert "error" in result
