"""End-to-end test: executor on iris classification."""
import json
from pathlib import Path
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from mlops_agents.contracts.training import TrainingPlan, TrainingPlanCandidate, TrialBudget
from mlops_agents.training.executor import run_training_plan


@pytest.fixture
def iris_csv(tmp_path):
    data = load_iris(as_frame=True)
    df = pd.concat([data.data, data.target.rename("target")], axis=1)
    p = tmp_path / "iris.csv"
    df.to_csv(p, index=False)
    return p


def test_executor_iris_classification_endtoend(iris_csv, tmp_path, monkeypatch):
    monkeypatch.setattr("mlops_agents.training.executor.settings.experience_pool_dir", tmp_path / "pool")
    plan = TrainingPlan(
        problem_type="classification",
        candidates=[
            TrainingPlanCandidate(priority=1, model_key="logistic_regression"),
            TrainingPlanCandidate(priority=2, model_key="random_forest_classifier"),
        ],
        trial_budget=TrialBudget(total_trials=10, min_trials_per_candidate=3, max_trials_per_candidate=10),
    )
    result = run_training_plan(
        plan=plan,
        processed_dataset_path=iris_csv,
        target_column="target",
        task_metadata={"problem_type": "classification", "target_column": "target"},
        output_dir=tmp_path / "splits",
        mlflow_experiment="test-iris",
        random_state=42,
    )
    assert Path(result.champion_model_path).exists()
    assert Path(result.train_pool_path).exists()
    assert Path(result.test_path).exists()
    assert Path(result.experience_record_path).exists()
    record = json.loads(Path(result.experience_record_path).read_text())
    assert record["problem_type"] == "classification"
    assert record["selected_solution"]["model_key"] in {"logistic_regression", "random_forest_classifier"}
    assert {c["model_key"] for c in record["models_tested"]} == {"logistic_regression", "random_forest_classifier"}
    assert result.champion_metrics["macro_f1"] > 0.85
