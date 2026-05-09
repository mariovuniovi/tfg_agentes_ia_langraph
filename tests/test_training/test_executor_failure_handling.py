"""Tests for executor failure handling: skip+retry+continue."""
import json
from pathlib import Path
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from mlops_agents.contracts.training import TrainingPlan, TrainingPlanCandidate, TrialBudget
from mlops_agents.training.executor import run_training_plan
from mlops_agents.models import factories as factories_module


@pytest.fixture
def iris_csv(tmp_path):
    data = load_iris(as_frame=True)
    df = pd.concat([data.data, data.target.rename("target")], axis=1)
    p = tmp_path / "iris.csv"
    df.to_csv(p, index=False)
    return p


def test_one_failed_one_succeeds_run_completes(iris_csv, tmp_path):
    """Force one factory to raise; other should still produce champion."""
    real = factories_module.FACTORY_REGISTRY["build_logistic_regression"]
    def boom(_params):
        raise RuntimeError("simulated failure")
    factories_module.FACTORY_REGISTRY["build_logistic_regression"] = boom
    try:
        plan = TrainingPlan(
            problem_type="classification",
            candidates=[
                TrainingPlanCandidate(priority=1, model_key="logistic_regression"),
                TrainingPlanCandidate(priority=2, model_key="random_forest_classifier"),
            ],
            trial_budget=TrialBudget(total_trials=6, min_trials_per_candidate=3, max_trials_per_candidate=3),
        )
        result = run_training_plan(
            plan=plan,
            processed_dataset_path=iris_csv,
            target_column="target",
            task_metadata={"problem_type": "classification", "target_column": "target"},
            output_dir=tmp_path / "splits",
            mlflow_experiment="test-failure",
            random_state=42,
        )
        record = json.loads(Path(result.experience_record_path).read_text())
        statuses = {r["model_key"]: r["status"] for r in record["models_tested"]}
        assert statuses["logistic_regression"] == "failed"
        assert statuses["random_forest_classifier"] == "successful"
        assert record["selected_solution"]["model_key"] == "random_forest_classifier"
    finally:
        factories_module.FACTORY_REGISTRY["build_logistic_regression"] = real
