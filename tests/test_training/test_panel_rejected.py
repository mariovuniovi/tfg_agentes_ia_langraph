"""Verify that multi-target panel forecasting is rejected at the executor entry."""
from pathlib import Path

import pandas as pd
import pytest

from mlops_agents.contracts.training import (
    ExogStrategySettings,
    ForecastingSettings,
    TrainingPlan,
    TrainingPlanCandidate,
    ValidationStrategy,
)
from mlops_agents.training.executor import run_training_plan


def test_panel_forecasting_raises_not_implemented(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "mlops_agents.training.executor.settings.experience_pool_dir",
        tmp_path / "pool",
    )
    # Build a minimal multi-target panel dataset
    df = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=60, freq="W").repeat(2),
        "series_id": ["A", "B"] * 60,
        "target": range(120),
    })
    csv = tmp_path / "panel.csv"
    df.to_csv(csv, index=False)

    plan = TrainingPlan(
        problem_type="forecasting",
        candidates=[TrainingPlanCandidate(priority=1, model_key="naive")],
        forecasting_settings=ForecastingSettings(
            validation_strategy=ValidationStrategy(horizon=4),
            exog_strategies=ExogStrategySettings(),
        ),
    )
    task_metadata = {
        "problem_type": "forecasting",
        "target_column": "target",
        "datetime_column": "date",
        "series_id_columns": ["series_id"],   # panel → should raise
        "frequency": "W",
        "forecast_horizon": 4,
    }
    with pytest.raises(NotImplementedError, match="panel|out of scope|series_id"):
        run_training_plan(
            plan=plan, processed_dataset_path=csv, target_column="target",
            task_metadata=task_metadata,
            output_dir=tmp_path / "out", mlflow_experiment="test_panel_reject",
        )
