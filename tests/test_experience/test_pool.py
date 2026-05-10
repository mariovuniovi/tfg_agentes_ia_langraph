"""Tests for ExperiencePool insert and query."""
import json
import sqlite3
import pytest
from mlops_agents.experience.pool import ExperiencePool
from mlops_agents.experience.schema import ExperienceRecord


def _sample_record(task_id: str = "iris_classification_2026-05-10_001") -> dict:
    return {
        "task_id": task_id,
        "problem_type": "classification",
        "dataset_name": "iris",
        "dataset_profile": {
            "schema_version": 1, "problem_type": "classification",
            "n_rows": "small", "n_features": "small", "missing_rate": "none",
            "n_categorical_features": "none", "n_numerical_features": "few",
            "n_classes": "small_multiclass", "class_balance": "balanced",
        },
        "training_plan_input": {"candidates": []},
        "split_artifacts": {
            "train_pool_path": "/tmp/iris_train_pool.csv",
            "test_path": "/tmp/iris_test.csv",
            "split_metadata_path": "/tmp/iris_split_metadata.json",
        },
        "mlflow": {"experiment_name": "test", "parent_run_id": "abc123"},
        "metric_to_optimize": "macro_f1", "metric_direction": "maximize",
        "candidate_selection_policy": {"primary": "best_validation_score", "tie_breaker": "complexity_rank", "tie_tolerance_relative": 0.01},
        "models_tested": [
            {"model_key": "logistic_regression", "status": "successful", "best_params": {"C": 1.0},
             "best_score": 0.94, "best_score_std": 0.02, "n_trials_used": 10, "duration_s": 3.5,
             "complexity_rank": 1, "mlflow_run_id": "run_lr"},
            {"model_key": "random_forest_classifier", "status": "failed", "error_type": "ValueError",
             "error_message": "boom", "n_trials_used": 0, "duration_s": 0.1, "complexity_rank": 2,
             "mlflow_run_id": "run_rf"},
        ],
        "selected_solution": {
            "model_key": "logistic_regression", "hyperparameters": {"C": 1.0},
            "validation_strategy": "stratified_5_fold_cv", "main_metric": "macro_f1",
            "validation_score": 0.94, "validation_std": 0.02, "complexity_rank": 1,
        },
        "experience_summary": "",
    }


def test_insert_from_record_writes_experiences_row(tmp_path):
    pool = ExperiencePool(tmp_path / "test.db")
    pool.insert_from_record(ExperienceRecord.model_validate(_sample_record()))
    assert pool.count() == 1


def test_insert_from_record_writes_candidate_rows(tmp_path):
    pool = ExperiencePool(tmp_path / "test.db")
    pool.insert_from_record(ExperienceRecord.model_validate(_sample_record()))
    conn = sqlite3.connect(str(tmp_path / "test.db"))
    rows = conn.execute("SELECT * FROM candidate_results WHERE task_id = ?",
                        ("iris_classification_2026-05-10_001",)).fetchall()
    conn.close()
    assert len(rows) == 2


def test_insert_writes_audit_json(tmp_path):
    audit_dir = tmp_path / "pool"
    pool = ExperiencePool(tmp_path / "test.db", audit_dir=audit_dir)
    pool.insert_from_record(ExperienceRecord.model_validate(_sample_record()))
    expected = audit_dir / "iris_classification_2026-05-10_001.json"
    assert expected.exists()
    assert json.loads(expected.read_text())["task_id"] == "iris_classification_2026-05-10_001"


def test_count_by_problem_type(tmp_path):
    pool = ExperiencePool(tmp_path / "test.db")
    pool.insert_from_record(ExperienceRecord.model_validate(_sample_record("a_classification_x_001")))
    pool.insert_from_record(ExperienceRecord.model_validate(
        {**_sample_record("b_regression_x_001"), "problem_type": "regression"}
    ))
    assert pool.count("classification") == 1
    assert pool.count("regression") == 1
    assert pool.count() == 2


def test_get_retrieves_record(tmp_path):
    pool = ExperiencePool(tmp_path / "test.db")
    pool.insert_from_record(ExperienceRecord.model_validate(_sample_record()))
    fetched = pool.get("iris_classification_2026-05-10_001")
    assert fetched.task_id == "iris_classification_2026-05-10_001"
    assert fetched.problem_type == "classification"
