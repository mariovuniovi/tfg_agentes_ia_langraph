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


def test_forecasting_reports_test_metrics_and_selection_score(air_passengers_csv, tmp_path, monkeypatch):
    monkeypatch.setattr("mlops_agents.training.executor.settings.experience_pool_dir", tmp_path / "pool")
    from mlops_agents.contracts.training import TrainingPlan, TrainingPlanCandidate, TrialBudget
    from mlops_agents.training.executor import run_training_plan

    plan = TrainingPlan(
        problem_type="forecasting",
        candidates=[TrainingPlanCandidate(priority=1, model_key="ets")],
        trial_budget=TrialBudget(total_trials=3, min_trials_per_candidate=3, max_trials_per_candidate=3),
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
        mlflow_experiment="test-air-honest",
        random_state=42,
    )
    assert "rmse" in result.champion_metrics
    assert "smape" in result.champion_metrics
    assert result.selection_score is not None
    assert result.forecast_chart_png is not None


def test_executor_fallback_resolves_multifold_validation(air_passengers_csv, tmp_path, monkeypatch):
    """A plan with forecasting_settings=None must route through resolve_validation_strategy
    (capacity-driven), NOT a hardcoded single_split. air_passengers (~144 obs) at horizon 12
    => multi-fold expanding_window."""
    monkeypatch.setattr("mlops_agents.training.executor.settings.experience_pool_dir", tmp_path / "pool")
    from mlops_agents.contracts.training import TrainingPlan, TrainingPlanCandidate, TrialBudget
    from mlops_agents.training.executor import run_training_plan

    plan = TrainingPlan(
        problem_type="forecasting",
        candidates=[TrainingPlanCandidate(priority=1, model_key="seasonal_naive")],
        trial_budget=TrialBudget(total_trials=2, min_trials_per_candidate=2, max_trials_per_candidate=2),
    )  # NOTE: no forecasting_settings -> exercises the executor fallback
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
        mlflow_experiment="test-fallback-multifold",
        random_state=42,
    )
    rec = json.loads(Path(result.experience_record_path).read_text())
    assert rec["validation_strategy"]["type"] == "expanding_window"   # not single_split — the fix
    assert rec["validation_strategy"]["n_folds"] >= 2
