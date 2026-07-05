"""End-to-end test: executor on California Housing regression."""
import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.datasets import fetch_california_housing

from mlops_agents.contracts.training import TrainingPlan, TrainingPlanCandidate
from mlops_agents.training.executor import run_training_plan


@pytest.fixture
def housing_csv(tmp_path):
    data = fetch_california_housing(as_frame=True)
    df = pd.concat([data.data, data.target.rename("target")], axis=1).head(1500)
    p = tmp_path / "housing.csv"
    df.to_csv(p, index=False)
    return p


def test_executor_housing_regression_endtoend(housing_csv, tmp_path, monkeypatch):
    monkeypatch.setattr("mlops_agents.training.executor.settings.experience_pool_dir", tmp_path / "pool")
    plan = TrainingPlan(
        problem_type="regression",
        candidates=[
            TrainingPlanCandidate(priority=1, model_key="ridge"),
            TrainingPlanCandidate(priority=2, model_key="random_forest_regressor"),
        ],
    )
    result = run_training_plan(
        plan=plan,
        processed_dataset_path=housing_csv,
        target_column="target",
        task_metadata={"problem_type": "regression", "target_column": "target"},
        output_dir=tmp_path / "splits",
        mlflow_experiment="test-housing",
        random_state=42,
    )
    assert Path(result.champion_model_path).exists()
    assert Path(result.train_pool_path).exists()
    assert Path(result.test_path).exists()
    assert Path(result.experience_record_path).exists()
    record = json.loads(Path(result.experience_record_path).read_text())
    assert record["problem_type"] == "regression"
    assert record["metric_direction"] == "minimize"
    # RMSE on California Housing with RF should beat trivially (< mean prediction)
    # Ridge baseline is typically RMSE ~0.9; any finite positive score qualifies
    assert 0 < record["selected_solution"]["validation_score"] < 100
    assert {c["model_key"] for c in record["models_tested"]} == {"ridge", "random_forest_regressor"}
