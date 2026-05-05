"""Unit tests for data tools (deterministic — no LLM calls)."""

import json
from pathlib import Path

import pandas as pd

from mlops_agents.tools.data_tools import (
    apply_column_mapping,
    check_missing_values,
    impute_missing_values,
    load_dataset,
    merge_datasets,
    parse_datetime_column,
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
    result = json.loads(validate_against_schema.invoke({
        "canonical_path": str(canonical_iris_csv),
        "schema_path": str(iris_schema_file),
    }))
    assert result["passed"] is True
    assert result["violations"] == []


def test_validate_against_schema_detects_nullable_violation(tmp_path, iris_schema_file):
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
    mapping = json.dumps({"feature_1": "sepal_length", "target": "target"})
    output_path = str(tmp_path / "out.csv")
    apply_column_mapping.invoke({
        "raw_path": str(sample_csv),
        "mapping_json": mapping,
        "output_path": output_path,
    })
    assert Path(output_path).exists()


def test_apply_column_mapping_reports_mapped_columns(tmp_path, sample_csv):
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
# impute_missing_values — role-aware dispatcher
# ---------------------------------------------------------------------------

import pytest


def test_impute_invalid_problem_type_raises(tmp_path):
    df = pd.DataFrame({"val": [1.0, None], "target": ["a", "b"]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="problem_type"):
        impute_missing_values.invoke({
            "dataset_path": str(path),
            "problem_type": "unknown",
            "target_column": "target",
        })


def test_impute_tabular_numeric_uses_mean(tmp_path):
    df = pd.DataFrame({"val": [1.0, None, 3.0], "target": ["a", "b", "c"]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    result = json.loads(impute_missing_values.invoke({
        "dataset_path": str(path),
        "problem_type": "classification",
        "target_column": "target",
    }))
    assert "val" in result["columns_imputed"]
    assert result["rows_affected"] == 1
    df_after = pd.read_csv(path)
    assert df_after["val"].isnull().sum() == 0
    assert abs(df_after["val"].iloc[1] - 2.0) < 0.01


def test_impute_tabular_categorical_uses_mode(tmp_path):
    df = pd.DataFrame({"species": ["setosa", "setosa", None], "target": [0, 1, 0]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    result = json.loads(impute_missing_values.invoke({
        "dataset_path": str(path),
        "problem_type": "regression",
        "target_column": "target",
    }))
    assert "species" in result["columns_imputed"]
    df_after = pd.read_csv(path)
    assert df_after["species"].iloc[2] == "setosa"


def test_impute_tabular_does_not_impute_target(tmp_path):
    # target column should NEVER be imputed — rows with missing target are dropped
    df = pd.DataFrame({"val": [1.0, 2.0, 3.0], "target": ["a", None, "c"]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    result = json.loads(impute_missing_values.invoke({
        "dataset_path": str(path),
        "problem_type": "classification",
        "target_column": "target",
    }))
    assert "target" not in result["columns_imputed"]
    df_after = pd.read_csv(path)
    assert df_after["target"].isnull().sum() == 0  # row was dropped, not filled
    assert len(df_after) == 2  # one row dropped


def test_impute_tabular_missing_target_reported_in_warnings(tmp_path):
    df = pd.DataFrame({"val": [1.0, 2.0], "target": [None, "b"]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    result = json.loads(impute_missing_values.invoke({
        "dataset_path": str(path),
        "problem_type": "classification",
        "target_column": "target",
    }))
    assert len(result["warnings"]) > 0
    assert "Dropped" in result["warnings"][0]


def test_impute_tabular_no_missing_is_noop(tmp_path):
    df = pd.DataFrame({"val": [1.0, 2.0], "target": [0, 1]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    result = json.loads(impute_missing_values.invoke({
        "dataset_path": str(path),
        "problem_type": "classification",
        "target_column": "target",
    }))
    assert result["columns_imputed"] == []
    assert result["rows_affected"] == 0


def test_impute_tabular_returns_output_path(tmp_path):
    df = pd.DataFrame({"val": [1.0, None], "target": [0, 1]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    result = json.loads(impute_missing_values.invoke({
        "dataset_path": str(path),
        "problem_type": "classification",
        "target_column": "target",
    }))
    assert "output_path" in result


def test_impute_tabular_writes_to_output_path(tmp_path):
    df = pd.DataFrame({"val": [1.0, None], "target": [0, 1]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    out = tmp_path / "out.csv"
    impute_missing_values.invoke({
        "dataset_path": str(path),
        "problem_type": "classification",
        "target_column": "target",
        "output_path": str(out),
    })
    assert out.exists()


def test_impute_forecasting_raises_if_datetime_null(tmp_path):
    df = pd.DataFrame({
        "date": [None, "2024-01-02", "2024-01-03"],
        "target": [1.0, 2.0, 3.0],
        "sid": ["s1", "s1", "s1"],
    })
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="null values"):
        impute_missing_values.invoke({
            "dataset_path": str(path),
            "problem_type": "forecasting",
            "target_column": "target",
            "datetime_column": "date",
            "series_id_columns": ["sid"],
        })


def test_impute_forecasting_raises_if_series_id_null(tmp_path):
    df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "target": [1.0, 2.0, 3.0],
        "sid": [None, "s1", "s1"],
    })
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="null values"):
        impute_missing_values.invoke({
            "dataset_path": str(path),
            "problem_type": "forecasting",
            "target_column": "target",
            "datetime_column": "date",
            "series_id_columns": ["sid"],
        })


def test_impute_forecasting_short_gap_target_filled(tmp_path):
    # Internal gap of 2 <= max_interpolation_gap=3 -> all filled
    df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"],
        "target": [1.0, None, None, 4.0, 5.0],
        "sid": ["s1"] * 5,
    })
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    result = json.loads(impute_missing_values.invoke({
        "dataset_path": str(path),
        "problem_type": "forecasting",
        "target_column": "target",
        "datetime_column": "date",
        "series_id_columns": ["sid"],
        "max_interpolation_gap": 3,
    }))
    assert result["target_large_gaps"] == []
    df_after = pd.read_csv(path)
    assert df_after["target"].isnull().sum() == 0


def test_impute_forecasting_large_gap_fully_preserved(tmp_path):
    # Internal gap of 5 > max_interpolation_gap=3 -> NONE of the 5 NaNs are filled
    df = pd.DataFrame({
        "date": ["2024-01-0" + str(i) for i in range(1, 9)],
        "target": [1.0, None, None, None, None, None, 7.0, 8.0],
        "sid": ["s1"] * 8,
    })
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    result = json.loads(impute_missing_values.invoke({
        "dataset_path": str(path),
        "problem_type": "forecasting",
        "target_column": "target",
        "datetime_column": "date",
        "series_id_columns": ["sid"],
        "max_interpolation_gap": 3,
    }))
    assert len(result["target_large_gaps"]) > 0
    assert result["target_large_gaps"][0]["gap_sizes"] == [5]
    df_after = pd.read_csv(path)
    # All 5 NaNs must remain — no partial imputation of large gaps
    assert df_after["target"].isnull().sum() == 5


def test_impute_forecasting_exogenous_numeric_forward_filled(tmp_path):
    df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "target": [1.0, 2.0, 3.0],
        "exog": [10.0, None, 30.0],
        "sid": ["s1"] * 3,
    })
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    result = json.loads(impute_missing_values.invoke({
        "dataset_path": str(path),
        "problem_type": "forecasting",
        "target_column": "target",
        "datetime_column": "date",
        "series_id_columns": ["sid"],
    }))
    assert "exog" in result["columns_imputed"]
    df_after = pd.read_csv(path)
    assert df_after["exog"].isnull().sum() == 0


def test_impute_forecasting_exogenous_categorical_filled(tmp_path):
    df = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "target": [1.0, 2.0, 3.0],
        "category": ["A", None, "A"],
        "sid": ["s1"] * 3,
    })
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    result = json.loads(impute_missing_values.invoke({
        "dataset_path": str(path),
        "problem_type": "forecasting",
        "target_column": "target",
        "datetime_column": "date",
        "series_id_columns": ["sid"],
    }))
    assert "category" in result["columns_imputed"]
    df_after = pd.read_csv(path)
    assert df_after["category"].isnull().sum() == 0


def test_impute_forecasting_multi_series_no_index_collision(tmp_path):
    """Multi-series concat must reset index so get_loc in gap helper never sees duplicates."""
    csv = tmp_path / "multi.csv"
    pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-01", "2024-01-02", "2024-01-03"],
        "sid": ["s1", "s1", "s1", "s2", "s2", "s2"],
        "sales": [1.0, None, 3.0, 4.0, None, 6.0],
    }).to_csv(csv, index=False)
    result = json.loads(impute_missing_values.invoke({
        "dataset_path": str(csv),
        "problem_type": "forecasting",
        "target_column": "sales",
        "datetime_column": "date",
        "series_id_columns": ["sid"],
        "max_interpolation_gap": 3,
    }))
    out = pd.read_csv(result["output_path"])
    assert out["sales"].isnull().sum() == 0
    assert len(out) == 6


def test_impute_file_not_found_returns_error():
    result = json.loads(impute_missing_values.invoke({
        "dataset_path": "/nonexistent/file.csv",
        "problem_type": "classification",
        "target_column": "target",
    }))
    assert "error" in result


# ---------------------------------------------------------------------------
# parse_datetime_column
# ---------------------------------------------------------------------------

def test_parse_datetime_column_parses_and_sorts(tmp_path):
    df = pd.DataFrame({
        "date": ["2024-01-03", "2024-01-01", "2024-01-02"],
        "val": [3, 1, 2],
    })
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    result = json.loads(parse_datetime_column.invoke({
        "dataset_path": str(path),
        "datetime_col": "date",
    }))
    assert result["null_count"] == 0
    assert result["dtype"] == "datetime64"
    assert result["output_path"] == str(path)  # overwrites input when output_path omitted
    df_after = pd.read_csv(path, parse_dates=["date"])
    assert list(df_after["val"]) == [1, 2, 3]


def test_parse_datetime_column_sorts_by_series_then_date(tmp_path):
    df = pd.DataFrame({
        "date": ["2024-01-02", "2024-01-01", "2024-01-02", "2024-01-01"],
        "store_id": ["B", "A", "A", "B"],
        "sales": [20, 10, 30, 40],
    })
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    parse_datetime_column.invoke({
        "dataset_path": str(path),
        "datetime_col": "date",
        "series_id_cols": ["store_id"],
    })
    df_after = pd.read_csv(path)
    # Should be sorted: A 2024-01-01, A 2024-01-02, B 2024-01-01, B 2024-01-02
    assert list(df_after["store_id"]) == ["A", "A", "B", "B"]
    assert list(df_after["sales"]) == [10, 30, 40, 20]


def test_parse_datetime_column_raises_if_nulls(tmp_path):
    df = pd.DataFrame({"date": [None, "2024-01-02"], "val": [1, 2]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="null"):
        parse_datetime_column.invoke({
            "dataset_path": str(path),
            "datetime_col": "date",
        })


def test_parse_datetime_column_raises_if_unparseable(tmp_path):
    df = pd.DataFrame({"date": ["not-a-date", "2024-01-02"], "val": [1, 2]})
    path = tmp_path / "data.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="null|unparseable"):
        parse_datetime_column.invoke({
            "dataset_path": str(path),
            "datetime_col": "date",
        })


def test_parse_datetime_column_writes_to_output_path(tmp_path):
    df = pd.DataFrame({"date": ["2024-01-01", "2024-01-02"], "val": [1, 2]})
    path = tmp_path / "data.csv"
    out = tmp_path / "sorted.csv"
    df.to_csv(path, index=False)
    parse_datetime_column.invoke({
        "dataset_path": str(path),
        "datetime_col": "date",
        "output_path": str(out),
    })
    assert out.exists()


def test_parse_datetime_column_returns_error_for_missing_file():
    result = json.loads(parse_datetime_column.invoke({
        "dataset_path": "/nonexistent/data.csv",
        "datetime_col": "date",
    }))
    assert "error" in result
