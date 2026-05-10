"""Factory functions for every registered model.

Each factory takes a hyperparameter dict (and for forecasting, task_metadata)
and returns a sklearn-compatible estimator or forecaster. Factories are referenced
by string name from registry.yaml.
"""

from __future__ import annotations

from typing import Any, Callable

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge


def build_logistic_regression(params: dict[str, Any]):
    return LogisticRegression(**params)


def build_random_forest_classifier(params: dict[str, Any]):
    return RandomForestClassifier(**params)


def build_lightgbm_classifier(params: dict[str, Any]):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(**{**params, "verbosity": -1})


def build_xgboost_classifier(params: dict[str, Any]):
    from xgboost import XGBClassifier
    return XGBClassifier(**{**params, "verbosity": 0})


def build_catboost_classifier(params: dict[str, Any]):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(**{**params, "verbose": 0})


def build_ridge(params: dict[str, Any]):
    return Ridge(**params)


def build_random_forest_regressor(params: dict[str, Any]):
    return RandomForestRegressor(**params)


def build_lightgbm_regressor(params: dict[str, Any]):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(**{**params, "verbosity": -1})


def build_xgboost_regressor(params: dict[str, Any]):
    from xgboost import XGBRegressor
    return XGBRegressor(**{**params, "verbosity": 0})


def build_catboost_regressor(params: dict[str, Any]):
    from catboost import CatBoostRegressor
    return CatBoostRegressor(**{**params, "verbose": 0})


_FREQ_TO_SEASON = {"H": 24, "D": 7, "W": 52, "MS": 12, "M": 12, "QS": 4, "YS": 1}


def _default_season_length(freq: str) -> int:
    return _FREQ_TO_SEASON.get(freq, 1)


def build_naive(spec: dict[str, Any]):
    """Naive forecaster: predicts the last observed value."""
    from statsforecast import StatsForecast
    from statsforecast.models import Naive
    freq = spec["task_metadata"]["frequency"]
    return StatsForecast(models=[Naive()], freq=freq, n_jobs=1)


def build_seasonal_naive(spec: dict[str, Any]):
    """Seasonal naive: predicts the value from one season ago."""
    from statsforecast import StatsForecast
    from statsforecast.models import SeasonalNaive
    freq = spec["task_metadata"]["frequency"]
    season_length = spec["params"].get("season_length", _default_season_length(freq))
    return StatsForecast(models=[SeasonalNaive(season_length=season_length)], freq=freq, n_jobs=1)


def build_ets(spec: dict[str, Any]):
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS
    freq = spec["task_metadata"]["frequency"]
    season_length = spec["params"].get("season_length", _default_season_length(freq))
    return StatsForecast(models=[AutoETS(season_length=season_length)], freq=freq, n_jobs=1)


def build_auto_arima(spec: dict[str, Any]):
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA
    freq = spec["task_metadata"]["frequency"]
    season_length = spec["params"].get("season_length", _default_season_length(freq))
    return StatsForecast(models=[AutoARIMA(season_length=season_length)], freq=freq, n_jobs=1)


def _split_lags_from_params(params: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Pull `lags` out of params; the rest goes to the regressor."""
    p = dict(params)
    lags = int(p.pop("lags", 12))
    return lags, p


def _wrap_with_skforecast(estimator: Any, lags: int) -> Any:
    from skforecast.recursive import ForecasterRecursiveMultiSeries
    return ForecasterRecursiveMultiSeries(estimator=estimator, lags=lags)


def build_random_forest_forecaster(spec: dict[str, Any]):
    lags, p = _split_lags_from_params(spec["params"])
    return _wrap_with_skforecast(RandomForestRegressor(**p), lags)


def build_extra_trees_forecaster(spec: dict[str, Any]):
    from sklearn.ensemble import ExtraTreesRegressor
    lags, p = _split_lags_from_params(spec["params"])
    return _wrap_with_skforecast(ExtraTreesRegressor(**p), lags)


def build_gbm_forecaster(spec: dict[str, Any]):
    from sklearn.ensemble import GradientBoostingRegressor
    lags, p = _split_lags_from_params(spec["params"])
    return _wrap_with_skforecast(GradientBoostingRegressor(**p), lags)


def build_lightgbm_forecaster(spec: dict[str, Any]):
    from lightgbm import LGBMRegressor
    lags, p = _split_lags_from_params(spec["params"])
    return _wrap_with_skforecast(LGBMRegressor(**{**p, "verbosity": -1}), lags)


def build_xgboost_forecaster(spec: dict[str, Any]):
    from xgboost import XGBRegressor
    lags, p = _split_lags_from_params(spec["params"])
    return _wrap_with_skforecast(XGBRegressor(**p), lags)


def build_svr_forecaster(spec: dict[str, Any]):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR
    lags, p = _split_lags_from_params(spec["params"])
    pipe = Pipeline([("scaler", StandardScaler()), ("svr", SVR(**p))])
    return _wrap_with_skforecast(pipe, lags)


FACTORY_REGISTRY: dict[str, Callable[..., Any]] = {
    "build_logistic_regression":      build_logistic_regression,
    "build_random_forest_classifier": build_random_forest_classifier,
    "build_lightgbm_classifier":      build_lightgbm_classifier,
    "build_xgboost_classifier":       build_xgboost_classifier,
    "build_catboost_classifier":      build_catboost_classifier,
    "build_ridge":                    build_ridge,
    "build_random_forest_regressor":  build_random_forest_regressor,
    "build_lightgbm_regressor":       build_lightgbm_regressor,
    "build_xgboost_regressor":        build_xgboost_regressor,
    "build_catboost_regressor":       build_catboost_regressor,
    "build_naive":                    build_naive,
    "build_seasonal_naive":           build_seasonal_naive,
    "build_ets":                      build_ets,
    "build_auto_arima":               build_auto_arima,
    "build_random_forest_forecaster": build_random_forest_forecaster,
    "build_extra_trees_forecaster":   build_extra_trees_forecaster,
    "build_gbm_forecaster":           build_gbm_forecaster,
    "build_lightgbm_forecaster":      build_lightgbm_forecaster,
    "build_xgboost_forecaster":       build_xgboost_forecaster,
    "build_svr_forecaster":           build_svr_forecaster,
}
