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
    TrialBudget,
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


# ---------------------------------------------------------------------------
# _narrow_seasonality_to_freq — restrict season_length grid to the data
# frequency (daily->7, weekly->52, ...) so seasonal models stay correct & fast.
# ---------------------------------------------------------------------------

from mlops_agents.models.loader import get_model  # noqa: E402
from mlops_agents.training.executor import _narrow_seasonality_to_freq  # noqa: E402


def _season_choices(spec):
    return list(spec.params["season_length"].choices)


def test_narrow_seasonality_daily_keeps_only_weekly_period():
    sp = get_model("auto_arima").search_space
    assert _season_choices(sp) == [4, 7, 12, 24, 52]  # full grid in registry
    narrowed = _narrow_seasonality_to_freq(sp, "D")
    assert _season_choices(narrowed) == [7]
    # original spec is not mutated
    assert _season_choices(sp) == [4, 7, 12, 24, 52]


def test_narrow_seasonality_weekly_keeps_yearly_period():
    sp = get_model("auto_arima").search_space
    assert _season_choices(_narrow_seasonality_to_freq(sp, "W")) == [52]


def test_narrow_seasonality_handles_pandas_alias_variants():
    sp = get_model("auto_arima").search_space
    # month-start / month-end / plain month all collapse to monthly -> 12
    assert _season_choices(_narrow_seasonality_to_freq(sp, "MS")) == [12]
    assert _season_choices(_narrow_seasonality_to_freq(sp, "ME")) == [12]
    # weekday-anchored weekly -> 52 (this is what the data actually carries)
    assert _season_choices(_narrow_seasonality_to_freq(sp, "W-MON")) == [52]
    # lowercase hourly -> 24 (emitted by the generators) — must NOT fall back to
    # the full grid, which would reintroduce the pathological m=52 hourly fit
    assert _season_choices(_narrow_seasonality_to_freq(sp, "h")) == [24]
    # yearly data has no intra-year seasonal cycle; enforce non-seasonal m=1 even
    # though the historical registry grid does not list 1.
    assert _season_choices(_narrow_seasonality_to_freq(sp, "YS")) == [1]


def test_narrow_seasonality_frequency_policy_wins_over_override():
    # Overrides are applied first by the executor. Frequency narrowing must still
    # be authoritative, otherwise a daily override like [52] reintroduces the
    # slow/wrong AutoARIMA fit we are trying to prevent.
    overridden = narrow_search_space(
        "auto_arima",
        {"season_length": SearchParamOverride(choices=[52])},
    )
    assert _season_choices(overridden) == [52]
    assert _season_choices(_narrow_seasonality_to_freq(overridden, "D")) == [7]


def test_narrow_seasonality_unknown_freq_keeps_full_grid():
    sp = get_model("ets").search_space
    assert _season_choices(_narrow_seasonality_to_freq(sp, None)) == [4, 7, 12, 24, 52]
    assert _season_choices(_narrow_seasonality_to_freq(sp, "weird")) == [4, 7, 12, 24, 52]


def test_narrow_seasonality_noop_when_no_season_length_param():
    # A model without a season_length categorical is returned unchanged.
    sp = get_model("random_forest_forecaster").search_space
    assert _narrow_seasonality_to_freq(sp, "D") is sp
