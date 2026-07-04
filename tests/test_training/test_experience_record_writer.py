"""Tests for the experience record writer."""
import json
from mlops_agents.training.experience_record import build_task_id, write_experience_record


def test_build_task_id_format():
    tid = build_task_id("iris", "classification", run_idx=1)
    assert tid.startswith("iris_classification_")
    assert tid.endswith("_001")


def test_write_experience_record_minimal(tmp_path):
    record = {
        "task_id": "iris_classification_2026-05-06_001",
        "problem_type": "classification",
        "dataset_profile": {"n_rows": "small"},
        "training_plan_input": {"candidates": []},
        "split_artifacts": {},
        "mlflow": {"experiment_name": "x", "parent_run_id": "abc"},
        "metric_to_optimize": "macro_f1",
        "metric_direction": "maximize",
        "candidate_selection_policy": {"primary": "best_validation_score"},
        "models_tested": [],
        "selected_solution": {},
        "experience_summary": "",
    }
    out_path = write_experience_record(record, tmp_path)
    assert out_path.exists()
    loaded = json.loads(out_path.read_text())
    assert loaded["task_id"] == record["task_id"]
