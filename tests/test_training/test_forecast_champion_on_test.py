import numpy as np
import pandas as pd

from mlops_agents.contracts.training import ExogStrategySettings, ForecastingSettings, ValidationStrategy
from mlops_agents.models.loader import get_model
from mlops_agents.training.executor import (
    _build_series_dict,
    _build_test_exog,
    _forecast_champion_on_test,
    _retrain_forecasting,
)


def _fs() -> ForecastingSettings:
    return ForecastingSettings(
        validation_strategy=ValidationStrategy(type="single_split", n_folds=1, horizon=3),
        exog_strategies=ExogStrategySettings(default_unknown_future="naive_carry"),
    )


def _task_meta() -> dict:
    return {
        "problem_type": "forecasting",
        "target_column": "y",
        "datetime_column": "ds",
        "series_id_columns": [],
        "frequency": "W",
        "forecast_horizon": 3,
        # _resolve_exog_availability reads `exogenous_columns` (list of {name, future_availability}).
        # 'holiday' is known-future (calendar); 'temp' is unknown-future.
        "exogenous_columns": [
            {"name": "temp", "future_availability": "unknown_future"},
            {"name": "holiday", "future_availability": "known_future"},
        ],
    }


def test_build_test_exog_extends_unknown_future_uses_actual_known_future():
    horizon = 3
    train = pd.DataFrame({
        "ds": pd.date_range("2023-01-02", periods=10, freq="W-MON"),
        "y": np.arange(10, dtype=float),
        "temp": np.arange(10, dtype=float),        # last value = 9.0
        "holiday": np.zeros(10, dtype=float),
    })
    test = pd.DataFrame({
        "ds": pd.date_range("2023-03-13", periods=horizon, freq="W-MON"),
        "y": [10.0, 11.0, 12.0],
        "temp": [100.0, 101.0, 102.0],             # very different from naive extension
        "holiday": [1.0, 0.0, 1.0],
    })
    series_dict = _build_series_dict(train, "ds", "y", [], "W")
    exog = _build_test_exog(train, test, _task_meta(), _fs(), horizon, "W", series_dict)

    # unknown-future temp must be the naive_carry extension (last train value 9.0), NOT the test actuals
    assert np.allclose(exog["temp"].to_numpy(), 9.0)
    assert not np.allclose(exog["temp"].to_numpy(), test["temp"].to_numpy())
    # known-future holiday uses the ACTUAL test values
    assert np.array_equal(exog["holiday"].to_numpy(), test["holiday"].to_numpy())


def test_forecast_champion_on_test_statsforecast(tmp_path):
    horizon = 4
    n = 40
    ds = pd.date_range("2023-01-02", periods=n, freq="W-MON")
    y = np.linspace(100, 200, n)  # clear ramp
    df = pd.DataFrame({"ds": ds, "y": y})
    train_pool = df.iloc[:-horizon].reset_index(drop=True)
    test_df = df.iloc[-horizon:].reset_index(drop=True)
    test_path = tmp_path / "test.csv"
    test_df.to_csv(test_path, index=False)

    task_meta = {
        "problem_type": "forecasting", "target_column": "y", "datetime_column": "ds",
        "series_id_columns": [], "frequency": "W", "forecast_horizon": horizon,
    }
    champion = {"model_key": "ets", "best_params": {"season_length": 1}, "best_score": 1.23}
    spec = get_model("ets")
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    champ_path = _retrain_forecasting(spec, champion, train_pool, task_meta, models_dir)

    fs = ForecastingSettings(
        validation_strategy=ValidationStrategy(type="single_split", n_folds=1, horizon=horizon),
        exog_strategies=ExogStrategySettings(),
    )
    metrics, preview = _forecast_champion_on_test(
        champion, champ_path, train_pool, test_path, task_meta, fs, "rmse",
    )
    assert "rmse" in metrics and metrics["rmse"] >= 0.0
    assert len(preview) == horizon
    assert set(preview[0].keys()) == {"ds", "y_true", "y_pred"}
    # y_true in the preview matches the held-out test target
    assert np.allclose([p["y_true"] for p in preview], test_df["y"].to_numpy())
