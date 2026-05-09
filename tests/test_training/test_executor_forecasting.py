"""End-to-end tests: executor on forecasting datasets."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from mlops_agents.contracts.training import TrainingPlan, TrainingPlanCandidate, TrialBudget
from mlops_agents.training.executor import run_training_plan


@pytest.fixture
def air_passengers_csv(tmp_path):
    dates = pd.date_range("2010-01-01", periods=144, freq="MS")
    rng = np.random.default_rng(0)
    trend = np.arange(144) * 1.5
    seasonal = 30 * np.sin(np.arange(144) * 2 * np.pi / 12)
    noise = rng.normal(scale=5.0, size=144)
    df = pd.DataFrame({"month": dates, "passengers": 200 + trend + seasonal + noise})
    p = tmp_path / "air_passengers.csv"
    df.to_csv(p, index=False)
    return p


def test_executor_forecasting_single_series_statistical(air_passengers_csv, tmp_path, monkeypatch):
    monkeypatch.setattr("mlops_agents.training.executor.settings.experience_pool_dir", tmp_path / "pool")
    plan = TrainingPlan(
        problem_type="forecasting",
        candidates=[
            TrainingPlanCandidate(priority=1, model_key="seasonal_naive"),
            TrainingPlanCandidate(priority=2, model_key="ets"),
        ],
        trial_budget=TrialBudget(total_trials=6, min_trials_per_candidate=3, max_trials_per_candidate=5),
    )
    result = run_training_plan(
        plan=plan,
        processed_dataset_path=air_passengers_csv,
        target_column="passengers",
        task_metadata={
            "problem_type": "forecasting", "target_column": "passengers",
            "datetime_column": "month", "series_id_columns": [],
            "frequency": "MS", "forecast_horizon": 12,
        },
        output_dir=tmp_path / "splits",
        mlflow_experiment="test-air",
        random_state=42,
    )
    record = json.loads(Path(result.experience_record_path).read_text())
    assert record["problem_type"] == "forecasting"
    assert record["selected_solution"]["model_key"] in {"seasonal_naive", "ets"}


def test_executor_forecasting_multi_series_supervised(tmp_path, monkeypatch):
    monkeypatch.setattr("mlops_agents.training.executor.settings.experience_pool_dir", tmp_path / "pool")
    rows = []
    rng = np.random.default_rng(0)
    for sid in ["a", "b", "c"]:
        dates = pd.date_range("2010-01-01", periods=120, freq="MS")
        offset = {"a": 0, "b": 50, "c": 100}[sid]
        vals = offset + np.arange(120) * 0.5 + 20 * np.sin(np.arange(120) * 2 * np.pi / 12) + rng.normal(scale=3, size=120)
        for d, v in zip(dates, vals):
            rows.append({"sid": sid, "ds": d, "y": float(v)})
    df = pd.DataFrame(rows)
    csv = tmp_path / "panel.csv"
    df.to_csv(csv, index=False)
    plan = TrainingPlan(
        problem_type="forecasting",
        candidates=[
            TrainingPlanCandidate(priority=1, model_key="seasonal_naive"),
            TrainingPlanCandidate(priority=2, model_key="lightgbm_forecaster"),
        ],
        trial_budget=TrialBudget(total_trials=6, min_trials_per_candidate=3, max_trials_per_candidate=5),
    )
    result = run_training_plan(
        plan=plan,
        processed_dataset_path=csv,
        target_column="y",
        task_metadata={
            "problem_type": "forecasting", "target_column": "y",
            "datetime_column": "ds", "series_id_columns": ["sid"],
            "frequency": "MS", "forecast_horizon": 12,
        },
        output_dir=tmp_path / "splits",
        mlflow_experiment="test-panel",
        random_state=42,
    )
    record = json.loads(Path(result.experience_record_path).read_text())
    assert record["selected_solution"]["model_key"] in {"seasonal_naive", "lightgbm_forecaster"}
