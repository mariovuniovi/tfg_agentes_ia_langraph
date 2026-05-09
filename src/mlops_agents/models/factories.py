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
    return LGBMClassifier(**params)


def build_xgboost_classifier(params: dict[str, Any]):
    from xgboost import XGBClassifier
    return XGBClassifier(**params)


def build_catboost_classifier(params: dict[str, Any]):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(**params)


def build_ridge(params: dict[str, Any]):
    return Ridge(**params)


def build_random_forest_regressor(params: dict[str, Any]):
    return RandomForestRegressor(**params)


def build_lightgbm_regressor(params: dict[str, Any]):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(**params)


def build_xgboost_regressor(params: dict[str, Any]):
    from xgboost import XGBRegressor
    return XGBRegressor(**params)


def build_catboost_regressor(params: dict[str, Any]):
    from catboost import CatBoostRegressor
    return CatBoostRegressor(**params)


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
}
