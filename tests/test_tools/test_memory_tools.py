"""Tests for memory retrieval tools."""
import json
from mlops_agents.tools.memory_tools import retrieve_similar_experiences, retrieve_ml_knowledge


def _cls_profile_json(n_rows: str = "medium") -> str:
    return json.dumps({
        "schema_version": 1, "problem_type": "classification",
        "n_rows": n_rows, "n_features": "small",
        "missing_rate": "none", "n_categorical_features": "none",
        "n_numerical_features": "few",
        "n_classes": "binary", "class_balance": "balanced",
    })


def test_retrieve_similar_experiences_returns_json(tmp_path, monkeypatch):
    monkeypatch.setattr("mlops_agents.tools.memory_tools.settings.experience_db_path", tmp_path / "db.db")
    result = retrieve_similar_experiences.invoke({
        "dataset_profile_json": _cls_profile_json(),
        "problem_type": "classification",
        "k": 5,
    })
    data = json.loads(result)
    assert isinstance(data, list)  # empty pool → empty list


def test_retrieve_similar_experiences_round_trip_pydantic(tmp_path, monkeypatch):
    from mlops_agents.experience.pool import ExperiencePool
    from mlops_agents.experience.schema import ExperienceRecord
    db = tmp_path / "db.db"
    monkeypatch.setattr("mlops_agents.tools.memory_tools.settings.experience_db_path", db)
    pool = ExperiencePool(db)
    pool.insert_from_record(ExperienceRecord.model_validate({
        "task_id": "test_cls_001",
        "problem_type": "classification",
        "dataset_name": "test",
        "dataset_profile": {
            "schema_version": 1, "problem_type": "classification",
            "n_rows": "medium", "n_features": "small",
            "missing_rate": "none", "n_categorical_features": "none",
            "n_numerical_features": "few",
            "n_classes": "binary", "class_balance": "balanced",
        },
        "training_plan_input": {}, "split_artifacts": {}, "mlflow": {},
        "metric_to_optimize": "macro_f1", "metric_direction": "maximize",
        "candidate_selection_policy": {},
        "models_tested": [{"model_key": "logistic_regression", "status": "successful",
                           "best_score": 0.93, "complexity_rank": 1, "n_trials_used": 5, "duration_s": 1.0}],
        "selected_solution": {"model_key": "logistic_regression",
                              "validation_score": 0.93, "complexity_rank": 1},
    }))
    result = retrieve_similar_experiences.invoke({
        "dataset_profile_json": _cls_profile_json("medium"),
        "problem_type": "classification",
        "k": 5,
    })
    data = json.loads(result)
    assert len(data) == 1
    assert data[0]["task_id"] == "test_cls_001"


def test_retrieve_ml_knowledge_returns_json():
    result = retrieve_ml_knowledge.invoke({
        "dataset_profile_json": _cls_profile_json("very_small"),
        "problem_type": "classification",
    })
    data = json.loads(result)
    assert isinstance(data, list)
    assert "classification_very_small_prefers_simple_models" in [r["rule_id"] for r in data]


def test_retrieve_ml_knowledge_returns_list_for_no_match():
    profile_json = json.dumps({
        "schema_version": 1, "problem_type": "regression",
        "n_rows": "medium", "n_features": "medium",
        "missing_rate": "none", "n_categorical_features": "none",
        "n_numerical_features": "many",
        "target_distribution": "near_normal",
    })
    result = retrieve_ml_knowledge.invoke({
        "dataset_profile_json": profile_json,
        "problem_type": "regression",
    })
    data = json.loads(result)
    assert isinstance(data, list)
