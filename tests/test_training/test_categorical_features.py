"""Tabular training must handle string/categorical features and drop id columns.

Regression for the gap surfaced by the retail partial-join dataset: sklearn models
(Ridge/RandomForest) cannot ingest string columns, and unique id columns must be
excluded from the feature matrix.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlops_agents.contracts.training import (
    TrainingPlan, TrainingPlanCandidate, TrialBudget,
)


def _reg_plan() -> TrainingPlan:
    return TrainingPlan(
        problem_type="regression",
        metric_to_optimize="rmse",
        candidates=[
            TrainingPlanCandidate(priority=1, model_key="ridge"),
            TrainingPlanCandidate(priority=2, model_key="random_forest_regressor"),
        ],
        trial_budget=TrialBudget(
            total_trials=4, allocation_strategy="equal",
            min_trials_per_candidate=2, max_trials_per_candidate=2,
        ),
    )


@pytest.fixture()
def categorical_regression_csv(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    n = 120
    region = rng.choice(["north", "south", "east", "west"], n)
    store_type = rng.choice(["flagship", "standard", "express"], n)
    size = rng.uniform(100, 2000, n)
    # target depends on the numeric + categorical signal so models can learn something
    base = {"north": 100, "south": 200, "east": 300, "west": 400}
    target = np.array([base[r] for r in region]) + 0.1 * size + rng.normal(0, 10, n)
    df = pd.DataFrame({
        "row_id": range(n),          # unique id → must be dropped from features
        "region": region,            # string categorical
        "store_type": store_type,    # string categorical
        "size": size,                # continuous numeric (high uniqueness, must be KEPT)
        "revenue": target,           # target
    })
    csv = tmp_path / "cat_reg.csv"
    df.to_csv(csv, index=False)
    return csv


def test_tabular_training_handles_categorical_and_drops_ids(
    categorical_regression_csv, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "mlops_agents.training.executor.settings.experience_pool_dir", tmp_path / "pool"
    )
    from mlops_agents.training.executor import run_training_plan

    result = run_training_plan(
        plan=_reg_plan(),
        processed_dataset_path=categorical_regression_csv,
        target_column="revenue",
        task_metadata={"problem_type": "regression", "target_column": "revenue", "id_columns": ["row_id"]},
        output_dir=tmp_path / "out",
        mlflow_experiment="test_categorical",
    )

    assert result.champion_metrics.get("rmse") is not None, "champion must produce RMSE"
    assert Path(result.champion_model_path).exists(), "champion model must be saved"


def test_all_numeric_dataset_still_trains(tmp_path, monkeypatch):
    """Guard: the categorical wrapper must not break all-numeric datasets."""
    monkeypatch.setattr(
        "mlops_agents.training.executor.settings.experience_pool_dir", tmp_path / "pool"
    )
    rng = np.random.default_rng(1)
    n = 120
    df = pd.DataFrame({
        "f1": rng.uniform(0, 1, n),
        "f2": rng.uniform(0, 10, n),
        "y": rng.uniform(0, 100, n),
    })
    csv = tmp_path / "num.csv"
    df.to_csv(csv, index=False)

    from mlops_agents.training.executor import run_training_plan

    result = run_training_plan(
        plan=_reg_plan(),
        processed_dataset_path=csv,
        target_column="y",
        task_metadata={"problem_type": "regression", "target_column": "y"},  # no id_columns
        output_dir=tmp_path / "out",
        mlflow_experiment="test_numeric",
    )
    assert result.champion_metrics.get("rmse") is not None
