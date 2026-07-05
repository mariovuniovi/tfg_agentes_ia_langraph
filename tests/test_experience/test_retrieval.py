"""Tests for weighted-overlap experience retrieval."""
from mlops_agents.experience.pool import ExperiencePool
from mlops_agents.experience.schema import ExperienceRecord


def _cls_record(task_id: str, n_rows: str, score: float) -> dict:
    return {
        "task_id": task_id, "problem_type": "classification", "dataset_name": task_id,
        "dataset_profile": {
            "schema_version": 1, "problem_type": "classification",
            "n_rows": n_rows, "n_features": "small", "missing_rate": "none",
            "n_categorical_features": "none", "n_numerical_features": "few",
            "n_classes": "binary", "class_balance": "balanced",
        },
        "training_plan_input": {}, "split_artifacts": {}, "mlflow": {},
        "metric_to_optimize": "macro_f1", "metric_direction": "maximize",
        "candidate_selection_policy": {},
        "models_tested": [{"model_key": "logistic_regression", "status": "successful",
                           "best_score": score, "complexity_rank": 1, "n_trials_used": 5, "duration_s": 1.0}],
        "selected_solution": {"model_key": "logistic_regression", "validation_score": score, "complexity_rank": 1},
    }


def test_find_similar_returns_closest_profile(tmp_path):
    pool = ExperiencePool(tmp_path / "db.db")
    pool.insert_from_record(ExperienceRecord.model_validate(_cls_record("a", "small", 0.90)))
    pool.insert_from_record(ExperienceRecord.model_validate(_cls_record("b", "medium", 0.85)))
    views = pool.find_similar(
        {"problem_type": "classification", "n_rows": "small", "n_features": "small",
         "missing_rate": "none", "n_categorical_features": "none",
         "n_numerical_features": "few", "n_classes": "binary", "class_balance": "balanced"},
        problem_type="classification", k=5,
    )
    assert views[0].task_id == "a"


def test_find_similar_hard_filters_by_problem_type(tmp_path):
    pool = ExperiencePool(tmp_path / "db.db")
    pool.insert_from_record(ExperienceRecord.model_validate(_cls_record("cls1", "small", 0.9)))
    pool.insert_from_record(ExperienceRecord.model_validate(
        {**_cls_record("reg1", "small", 0.9), "problem_type": "regression"}
    ))
    views = pool.find_similar({"n_rows": "small"}, problem_type="classification", k=5)
    task_ids = {v.task_id for v in views}
    assert "cls1" in task_ids
    assert "reg1" not in task_ids


def test_find_similar_empty_pool_returns_empty(tmp_path):
    pool = ExperiencePool(tmp_path / "db.db")
    assert pool.find_similar({"n_rows": "small"}, "classification", k=5) == []


def test_similarity_ratio_is_normalized(tmp_path):
    pool = ExperiencePool(tmp_path / "db.db")
    pool.insert_from_record(ExperienceRecord.model_validate(_cls_record("x", "small", 0.9)))
    views = pool.find_similar(
        {"problem_type": "classification", "n_rows": "small", "n_features": "small",
         "missing_rate": "none", "n_categorical_features": "none",
         "n_numerical_features": "few", "n_classes": "binary", "class_balance": "balanced"},
        "classification", k=1,
    )
    assert 0.0 <= views[0].similarity_ratio <= 1.0


def test_find_similar_returns_at_most_k(tmp_path):
    pool = ExperiencePool(tmp_path / "db.db")
    for i in range(5):
        pool.insert_from_record(ExperienceRecord.model_validate(_cls_record(f"r{i}", "small", 0.9)))
    views = pool.find_similar({"n_rows": "small"}, "classification", k=3)
    assert len(views) <= 3
