"""Tests for select_validation_strategy and validate_forecasting_plan."""
import pytest

from mlops_agents.contracts.training import (
    ExogStrategySettings,
    ForecastingSettings,
    TrainingPlan,
    TrainingPlanCandidate,
    TrialBudget,
    ValidationStrategy,
)
from mlops_agents.contracts.profile import DatasetProfile
from mlops_agents.training.validation_policy import (
    select_validation_strategy,
    resolve_rolling_window_size,
    validate_forecasting_plan,
)


def _profile(history_length: str) -> DatasetProfile:
    return DatasetProfile(
        problem_type="forecasting",
        n_rows="medium",
        history_length=history_length,
        n_features="small",
        missing_rate="none",
        n_categorical_features="none",
        n_numerical_features="few",
    )


def _task_meta(horizon=6, exog_cols=None, expected_drift=None):
    meta = {
        "problem_type": "forecasting",
        "target_column": "y",
        "datetime_column": "date",
        "series_id_columns": [],
        "frequency": "W",
        "forecast_horizon": horizon,
    }
    if exog_cols is not None:
        meta["exogenous_columns"] = exog_cols
    if expected_drift is not None:
        meta["expected_drift"] = expected_drift
    return meta


# ─── select_validation_strategy ────────────────────────────────────

def test_short_history_returns_single_split_even_with_high_drift():
    s = select_validation_strategy(_profile("short"), _task_meta(expected_drift="high"))
    assert s.type == "single_split"
    assert s.n_folds == 1


def test_medium_history_low_drift_returns_expanding():
    s = select_validation_strategy(_profile("medium"), _task_meta())
    assert s.type == "expanding_window"
    assert s.n_folds == 3


def test_long_history_high_drift_returns_rolling():
    s = select_validation_strategy(_profile("long"), _task_meta(expected_drift="high"))
    assert s.type == "rolling_window"
    assert s.n_folds == 3
    assert s.window_size is None  # auto


# ─── resolve_rolling_window_size ───────────────────────────────────

def test_rolling_window_size_respects_floor_and_capacity():
    # 200 history, horizon 10, 3 folds → upper=170, base=max(30,50)=50, result=50
    w = resolve_rolling_window_size(total_history=200, horizon=10, n_folds=3, season_length=None)
    assert w == 50


def test_rolling_window_size_clamps_to_horizon_when_capacity_tight():
    # total=35, horizon=10, n_folds=3 → upper=5, max(upper, horizon)=10, min(base=50, 10)=10
    w = resolve_rolling_window_size(total_history=35, horizon=10, n_folds=3, season_length=None)
    assert w == 10


def test_rolling_window_size_ignores_season_length_in_mvp():
    w1 = resolve_rolling_window_size(total_history=200, horizon=10, n_folds=3, season_length=None)
    w2 = resolve_rolling_window_size(total_history=200, horizon=10, n_folds=3, season_length=52)
    assert w1 == w2


# ─── validate_forecasting_plan ─────────────────────────────────────

def _plan_with(forecasting_settings):
    return TrainingPlan(
        problem_type="forecasting",
        candidates=[TrainingPlanCandidate(priority=1, model_key="naive")],
        trial_budget=TrialBudget(total_trials=2, allocation_strategy="equal",
                                 min_trials_per_candidate=1, max_trials_per_candidate=2),
        forecasting_settings=forecasting_settings,
    )


def test_validate_raises_when_horizon_mismatch():
    fs = ForecastingSettings(
        validation_strategy=ValidationStrategy(horizon=12),  # task says 6
        exog_strategies=ExogStrategySettings(),
    )
    plan = _plan_with(fs)
    with pytest.raises(ValueError, match="horizon"):
        validate_forecasting_plan(
            plan, _task_meta(horizon=6), _profile("medium"),
            {"single_series": True, "series_lengths": None, "total_len": 200},
        )


def test_validate_raises_when_per_column_references_unknown_column():
    fs = ForecastingSettings(
        validation_strategy=ValidationStrategy(horizon=6),
        exog_strategies=ExogStrategySettings(per_column={"nope": "ets"}),
    )
    plan = _plan_with(fs)
    with pytest.raises(ValueError, match="unknown|not.*exogenous"):
        validate_forecasting_plan(
            plan,
            _task_meta(exog_cols=[{"name": "oil", "future_availability": "unknown_future"}]),
            _profile("medium"),
            {"single_series": True, "series_lengths": None, "total_len": 200},
        )


def test_validate_raises_when_overriding_known_future_column():
    fs = ForecastingSettings(
        validation_strategy=ValidationStrategy(horizon=6),
        exog_strategies=ExogStrategySettings(per_column={"holiday": "ets"}),
    )
    plan = _plan_with(fs)
    with pytest.raises(ValueError, match="known_future"):
        validate_forecasting_plan(
            plan,
            _task_meta(exog_cols=[{"name": "holiday", "future_availability": "known_future"}]),
            _profile("medium"),
            {"single_series": True, "series_lengths": None, "total_len": 200},
        )


def test_validate_raises_when_insufficient_history():
    fs = ForecastingSettings(
        validation_strategy=ValidationStrategy(type="expanding_window", n_folds=3, horizon=20, step_size=20),
        exog_strategies=ExogStrategySettings(),
    )
    plan = _plan_with(fs)
    with pytest.raises(ValueError, match="history|enough"):
        validate_forecasting_plan(
            plan, _task_meta(horizon=20), _profile("medium"),
            {"single_series": True, "series_lengths": None, "total_len": 50},
        )


def test_validate_panel_rejects_per_column_overrides():
    fs = ForecastingSettings(
        validation_strategy=ValidationStrategy(horizon=6),
        exog_strategies=ExogStrategySettings(per_column={"oil": "ets"}),
    )
    plan = _plan_with(fs)
    with pytest.raises(NotImplementedError, match="panel|multi-target"):
        validate_forecasting_plan(
            plan, _task_meta(exog_cols=[{"name": "oil", "future_availability": "unknown_future"}]),
            _profile("medium"),
            {"single_series": False, "series_lengths": {"A": 100, "B": 100}, "total_len": 200},
        )


# ─── resolve_validation_strategy ───────────────────────────────────

from mlops_agents.training.validation_policy import resolve_validation_strategy


def _vmeta(horizon=8, drift="low"):
    return {"forecast_horizon": horizon, "expected_drift": drift}


def test_resolve_ample_history_caps_at_5_folds():
    vs = resolve_validation_strategy(_vmeta(), n_obs=156)
    assert vs.type == "expanding_window"
    assert vs.n_folds == 5
    assert vs.horizon == 8


def test_resolve_moderate_history_two_folds():
    vs = resolve_validation_strategy(_vmeta(), n_obs=60)
    assert vs.type == "expanding_window"
    assert vs.n_folds == 2


def test_resolve_floor_is_single_split():
    vs = resolve_validation_strategy(_vmeta(), n_obs=46)
    assert vs.type == "single_split"
    assert vs.n_folds == 1


def test_resolve_too_small_raises():
    import pytest
    with pytest.raises(ValueError, match="need >= 46 observations for horizon 8, have 20"):
        resolve_validation_strategy(_vmeta(), n_obs=20)


def test_resolve_high_drift_uses_rolling_window():
    vs = resolve_validation_strategy(_vmeta(drift="high"), n_obs=156)
    assert vs.type == "rolling_window"
    assert vs.n_folds == 5
