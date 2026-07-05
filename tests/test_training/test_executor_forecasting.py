"""End-to-end tests: executor on forecasting datasets."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlops_agents.contracts.training import (
    SearchParamOverride,
    TrainingPlan,
    TrainingPlanCandidate,
)
from mlops_agents.training.executor import run_training_plan
from mlops_agents.training.override_validation import narrow_search_space


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
    from mlops_agents.contracts.training import TrainingPlan, TrainingPlanCandidate
    from mlops_agents.training.executor import run_training_plan

    plan = TrainingPlan(
        problem_type="forecasting",
        candidates=[TrainingPlanCandidate(priority=1, model_key="ets")],
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
    from mlops_agents.contracts.training import TrainingPlan, TrainingPlanCandidate
    from mlops_agents.training.executor import run_training_plan

    plan = TrainingPlan(
        problem_type="forecasting",
        candidates=[TrainingPlanCandidate(priority=1, model_key="seasonal_naive")],
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


# ---------------------------------------------------------------------------
# narrow_seasonality_to_freq — restrict season_length grid to the data
# frequency (daily->7, weekly->52, ...) so seasonal models stay correct & fast.
# ---------------------------------------------------------------------------

from mlops_agents.contracts.training import ValidationStrategy  # noqa: E402
from mlops_agents.models.loader import get_model  # noqa: E402
from mlops_agents.training.forecasting_runner import min_fold_train_len, narrow_seasonality_to_freq  # noqa: E402


def _season_choices(spec):
    return list(spec.params["season_length"].choices)


def test_narrow_seasonality_applies_model_grid_and_does_not_mutate():
    sp = get_model("auto_arima").search_space
    out = narrow_seasonality_to_freq(sp, "D", "auto_arima", 10000)
    assert _season_choices(out) == [1, 7]
    assert _season_choices(sp) == [4, 7, 12, 24, 52]  # original registry spec untouched


def test_narrow_seasonality_seasonal_naive_gets_rich_grid():
    sp = get_model("seasonal_naive").search_space
    out = narrow_seasonality_to_freq(sp, "W", "seasonal_naive", 10000)
    assert _season_choices(out) == [1, 4, 13, 52]


def test_narrow_seasonality_unknown_freq_keeps_full_grid():
    sp = get_model("ets").search_space
    out = narrow_seasonality_to_freq(sp, None, "ets", 1000)
    assert _season_choices(out) == [4, 7, 12, 24, 52]


def test_narrow_seasonality_frequency_policy_wins_over_override():
    overridden = narrow_search_space("auto_arima", {"season_length": SearchParamOverride(choices=[52])})
    assert _season_choices(overridden) == [52]
    # daily data: frequency grid replaces the (wrong) override
    assert _season_choices(narrow_seasonality_to_freq(overridden, "D", "auto_arima", 10000)) == [1, 7]


def test_narrow_seasonality_noop_when_no_season_length_param():
    sp = get_model("random_forest_forecaster").search_space
    assert narrow_seasonality_to_freq(sp, "D", "random_forest_forecaster", 10000) is sp


def test_min_fold_train_len_rolling_uses_window():
    vs = ValidationStrategy(type="rolling_window", n_folds=5, horizon=8, step_size=8, window_size=70)
    assert min_fold_train_len(vs, 1000) == 70


def test_min_fold_train_len_expanding_uses_first_fold():
    # first fold trains on train_pool_len - (k-1)*horizon
    vs = ValidationStrategy(type="expanding_window", n_folds=5, horizon=8, step_size=8)
    assert min_fold_train_len(vs, 200) == 200 - 4 * 8  # 168


def test_min_fold_train_len_single_split_uses_full_pool():
    vs = ValidationStrategy(type="single_split", n_folds=1, horizon=8)
    assert min_fold_train_len(vs, 200) == 200


def test_narrow_seasonality_prunes_season_unestimable_in_rolling_fold():
    # full pool (104 weekly rows) would keep 52, but each rolling fold sees only 70
    # rows -> 52 (needs >=104) must be dropped; 13 (>=26) and 4 kept.
    sp = get_model("seasonal_naive").search_space
    vs = ValidationStrategy(type="rolling_window", n_folds=5, horizon=8, step_size=8, window_size=70)
    prune_n = min_fold_train_len(vs, 104)
    out = narrow_seasonality_to_freq(sp, "W", "seasonal_naive", prune_n)
    assert _season_choices(out) == [1, 4, 13]
