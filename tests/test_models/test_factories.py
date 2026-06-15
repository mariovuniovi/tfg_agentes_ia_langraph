"""Unit tests for model factories."""

import numpy as np
import pandas as pd
import pytest

from mlops_agents.models.factories import FACTORY_REGISTRY


@pytest.fixture
def tabular_classification_xy():
    rng = np.random.default_rng(42)
    Xa = rng.normal(size=(60, 4))
    y = (Xa[:, 0] + Xa[:, 1] > 0).astype(int)
    # Factories wrap estimators in a sklearn Pipeline whose preprocessor selects
    # columns by dtype, so features must be a DataFrame (as in the executor).
    X = pd.DataFrame(Xa, columns=[f"f{i}" for i in range(4)])
    return X, y


def _check_classifier_fits_and_predicts(factory_name: str, params: dict, xy):
    X, y = xy
    factory = FACTORY_REGISTRY[factory_name]
    model = factory(params)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape
    assert set(np.unique(preds)).issubset({0, 1})


def test_logistic_regression_factory(tabular_classification_xy):
    _check_classifier_fits_and_predicts(
        "build_logistic_regression",
        {"C": 1.0, "penalty": "l2", "max_iter": 200},
        tabular_classification_xy,
    )


def test_random_forest_classifier_factory(tabular_classification_xy):
    _check_classifier_fits_and_predicts(
        "build_random_forest_classifier",
        {"n_estimators": 50, "max_depth": 5, "random_state": 0},
        tabular_classification_xy,
    )


def test_lightgbm_classifier_factory(tabular_classification_xy):
    _check_classifier_fits_and_predicts(
        "build_lightgbm_classifier",
        {"n_estimators": 50, "learning_rate": 0.1, "num_leaves": 31, "random_state": 0, "verbosity": -1},
        tabular_classification_xy,
    )


def test_xgboost_classifier_factory(tabular_classification_xy):
    _check_classifier_fits_and_predicts(
        "build_xgboost_classifier",
        {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 4, "random_state": 0,
         "tree_method": "hist", "verbosity": 0},
        tabular_classification_xy,
    )


def test_catboost_classifier_factory(tabular_classification_xy):
    _check_classifier_fits_and_predicts(
        "build_catboost_classifier",
        {"iterations": 50, "learning_rate": 0.1, "depth": 4, "random_seed": 0, "verbose": False},
        tabular_classification_xy,
    )


@pytest.fixture
def tabular_regression_xy():
    rng = np.random.default_rng(42)
    Xa = rng.normal(size=(60, 4))
    y = Xa[:, 0] + 0.5 * Xa[:, 1] + rng.normal(scale=0.1, size=60)
    X = pd.DataFrame(Xa, columns=[f"f{i}" for i in range(4)])
    return X, y


def _check_regressor_fits_and_predicts(factory_name: str, params: dict, xy):
    X, y = xy
    factory = FACTORY_REGISTRY[factory_name]
    model = factory(params)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == y.shape
    assert np.isfinite(preds).all()


def test_ridge_factory(tabular_regression_xy):
    _check_regressor_fits_and_predicts(
        "build_ridge",
        {"alpha": 1.0, "random_state": 0},
        tabular_regression_xy,
    )


def test_random_forest_regressor_factory(tabular_regression_xy):
    _check_regressor_fits_and_predicts(
        "build_random_forest_regressor",
        {"n_estimators": 50, "max_depth": 5, "random_state": 0},
        tabular_regression_xy,
    )


def test_lightgbm_regressor_factory(tabular_regression_xy):
    _check_regressor_fits_and_predicts(
        "build_lightgbm_regressor",
        {"n_estimators": 50, "learning_rate": 0.1, "num_leaves": 31, "random_state": 0, "verbosity": -1},
        tabular_regression_xy,
    )


def test_xgboost_regressor_factory(tabular_regression_xy):
    _check_regressor_fits_and_predicts(
        "build_xgboost_regressor",
        {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 4, "random_state": 0,
         "tree_method": "hist", "verbosity": 0},
        tabular_regression_xy,
    )


def test_catboost_regressor_factory(tabular_regression_xy):
    _check_regressor_fits_and_predicts(
        "build_catboost_regressor",
        {"iterations": 50, "learning_rate": 0.1, "depth": 4, "random_seed": 0, "verbose": False},
        tabular_regression_xy,
    )


@pytest.fixture
def panel_dataframe():
    """Multi-series panel data: 2 series x 36 monthly periods."""
    import pandas as pd
    dates = pd.date_range("2020-01-01", periods=36, freq="MS")
    rows = []
    for sid in ["s1", "s2"]:
        for i, d in enumerate(dates):
            rows.append({"unique_id": sid, "ds": d, "y": float(i) + (1.0 if sid == "s1" else 5.0)})
    return pd.DataFrame(rows)


def _check_stat_forecaster_fits_and_predicts(factory_name: str, params: dict, panel):
    factory = FACTORY_REGISTRY[factory_name]
    sf = factory({"task_metadata": {"frequency": "MS", "forecast_horizon": 6}, "params": params})
    sf.fit(panel)
    fcst = sf.predict(h=6)
    assert len(fcst) == 6 * 2     # 6 horizons x 2 series
    assert "unique_id" in fcst.columns and "ds" in fcst.columns


def test_naive_factory(panel_dataframe):
    _check_stat_forecaster_fits_and_predicts("build_naive", {}, panel_dataframe)


def test_seasonal_naive_factory(panel_dataframe):
    _check_stat_forecaster_fits_and_predicts(
        "build_seasonal_naive", {"season_length": 12}, panel_dataframe,
    )


def test_ets_factory(panel_dataframe):
    _check_stat_forecaster_fits_and_predicts(
        "build_ets", {"season_length": 12}, panel_dataframe,
    )


def test_auto_arima_factory(panel_dataframe):
    _check_stat_forecaster_fits_and_predicts(
        "build_auto_arima", {"season_length": 12}, panel_dataframe,
    )


def _check_supervised_forecaster_fits_and_predicts(factory_name: str, params: dict, panel):
    """skforecast wants a wide series_dict from the long panel."""
    factory = FACTORY_REGISTRY[factory_name]
    forecaster = factory({"task_metadata": {"forecast_horizon": 6}, "params": params})
    series_dict = {
        sid: g.set_index("ds")["y"].asfreq("MS")
        for sid, g in panel.groupby("unique_id")
    }
    forecaster.fit(series=series_dict)
    preds = forecaster.predict(steps=6)
    assert len(preds) == 6 * 2
    # skforecast returns long format with columns ['level', 'pred']
    assert "level" in preds.columns or "unique_id" in preds.columns


def test_random_forest_forecaster_factory(panel_dataframe):
    _check_supervised_forecaster_fits_and_predicts(
        "build_random_forest_forecaster",
        {"lags": 12, "n_estimators": 30, "max_depth": 5, "random_state": 0},
        panel_dataframe,
    )


def test_extra_trees_forecaster_factory(panel_dataframe):
    _check_supervised_forecaster_fits_and_predicts(
        "build_extra_trees_forecaster",
        {"lags": 12, "n_estimators": 30, "max_depth": 5, "random_state": 0},
        panel_dataframe,
    )


def test_gbm_forecaster_factory(panel_dataframe):
    _check_supervised_forecaster_fits_and_predicts(
        "build_gbm_forecaster",
        {"lags": 12, "n_estimators": 30, "max_depth": 3, "random_state": 0},
        panel_dataframe,
    )


def test_lightgbm_forecaster_factory(panel_dataframe):
    _check_supervised_forecaster_fits_and_predicts(
        "build_lightgbm_forecaster",
        {"lags": 12, "n_estimators": 30, "learning_rate": 0.1, "num_leaves": 15,
         "random_state": 0, "verbosity": -1},
        panel_dataframe,
    )


def test_xgboost_forecaster_factory(panel_dataframe):
    _check_supervised_forecaster_fits_and_predicts(
        "build_xgboost_forecaster",
        {"lags": 12, "n_estimators": 30, "learning_rate": 0.1, "max_depth": 3,
         "random_state": 0, "tree_method": "hist", "verbosity": 0},
        panel_dataframe,
    )


def test_svr_forecaster_factory(panel_dataframe):
    _check_supervised_forecaster_fits_and_predicts(
        "build_svr_forecaster",
        {"lags": 12, "C": 1.0, "epsilon": 0.1, "kernel": "rbf"},
        panel_dataframe,
    )
