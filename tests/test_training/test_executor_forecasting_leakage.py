"""End-to-end leakage and fold tests for the rewritten forecasting executor."""
import json
from pathlib import Path

import numpy as np
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


def _synthetic_csv(tmp_path: Path, rows: int = 200) -> Path:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2018-01-01", periods=rows, freq="W")
    oil = np.cumsum(rng.normal(0, 1, rows)) + 50
    holiday_flag = ((np.arange(rows) % 13) == 0).astype(int)
    target = 100 + 0.3 * oil + 5 * holiday_flag + rng.normal(0, 1, rows)
    df = pd.DataFrame({"date": dates, "target": target, "oil": oil, "holiday_flag": holiday_flag})
    p = tmp_path / "synth.csv"
    df.to_csv(p, index=False)
    return p


def _plan(horizon=10):
    return TrainingPlan(
        problem_type="forecasting",
        candidates=[TrainingPlanCandidate(priority=1, model_key="lightgbm_forecaster")],
        forecasting_settings=ForecastingSettings(
            validation_strategy=ValidationStrategy(horizon=horizon),
            exog_strategies=ExogStrategySettings(
                per_column={"oil": "naive_carry"},
                default_unknown_future="naive_carry",
            ),
        ),
    )


def _task_meta(horizon=10):
    return {
        "problem_type": "forecasting", "target_column": "target",
        "datetime_column": "date", "series_id_columns": [],
        "frequency": "W", "forecast_horizon": horizon,
        "exogenous_columns": [
            {"name": "oil", "future_availability": "unknown_future"},
            {"name": "holiday_flag", "future_availability": "known_future"},
        ],
    }


def test_unknown_future_exog_is_extended_not_leaked(tmp_path, monkeypatch):
    monkeypatch.setattr("mlops_agents.training.executor.settings.experience_pool_dir", tmp_path / "pool")
    csv = _synthetic_csv(tmp_path, 200)
    plan = _plan(horizon=10)
    result = run_training_plan(
        plan=plan, processed_dataset_path=csv, target_column="target",
        task_metadata=_task_meta(horizon=10),
        output_dir=tmp_path / "out", mlflow_experiment="test_leak",
    )
    assert result.champion_candidate is not None
    rec = json.loads(Path(result.experience_record_path).read_text())
    assert rec.get("exog_strategies", {}).get("oil") == "naive_carry"
    assert rec.get("exog_availability", {}).get("holiday_flag") == "known_future"


def test_k_fold_runs_three_folds(tmp_path, monkeypatch):
    monkeypatch.setattr("mlops_agents.training.executor.settings.experience_pool_dir", tmp_path / "pool")
    csv = _synthetic_csv(tmp_path, 400)
    plan = _plan(horizon=10)
    plan.forecasting_settings.validation_strategy = ValidationStrategy(
        type="expanding_window", n_folds=3, horizon=10, step_size=10,
    )
    result = run_training_plan(
        plan=plan, processed_dataset_path=csv, target_column="target",
        task_metadata=_task_meta(horizon=10),
        output_dir=tmp_path / "out", mlflow_experiment="test_kfold",
    )
    rec = json.loads(Path(result.experience_record_path).read_text())
    pfm = rec.get("per_fold_metrics") or []
    assert len(pfm) == 3


def test_plan_with_unknown_column_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("mlops_agents.training.executor.settings.experience_pool_dir", tmp_path / "pool")
    csv = _synthetic_csv(tmp_path, 200)
    plan = _plan(horizon=10)
    plan.forecasting_settings.exog_strategies.per_column = {"nonexistent": "ets"}
    with pytest.raises(ValueError, match="per_column|nonexistent"):
        run_training_plan(
            plan=plan, processed_dataset_path=csv, target_column="target",
            task_metadata=_task_meta(horizon=10),
            output_dir=tmp_path / "out", mlflow_experiment="test_invalid",
        )


def test_plan_without_exog_columns_treats_all_as_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr("mlops_agents.training.executor.settings.experience_pool_dir", tmp_path / "pool")
    csv = _synthetic_csv(tmp_path, 200)
    plan = _plan(horizon=10)
    plan.forecasting_settings.exog_strategies.per_column = {}
    meta = _task_meta(horizon=10)
    del meta["exogenous_columns"]
    result = run_training_plan(
        plan=plan, processed_dataset_path=csv, target_column="target",
        task_metadata=meta, output_dir=tmp_path / "out", mlflow_experiment="test_default",
    )
    rec = json.loads(Path(result.experience_record_path).read_text())
    avail = rec.get("exog_availability") or {}
    assert avail.get("oil") == "unknown_future"
    assert avail.get("holiday_flag") == "unknown_future"
