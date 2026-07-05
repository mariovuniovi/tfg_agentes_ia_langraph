"""Factory functions for every registered model.

Each factory takes a hyperparameter dict (and for forecasting, task_metadata)
and returns a sklearn-compatible estimator or forecaster. Factories are referenced
by string name from registry.yaml.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from mlops_agents.forecasting.seasonality import default_season_length

_ARIMA_APPROX_MIN_OBS = 500  # above this, exact ARIMA order search is too slow -> use CSS approximation


def arima_use_approximation(n_obs: int) -> bool:
    """Use AutoARIMA's fast CSS order search only when the series is long enough that
    exact search is too slow AND CSS order selection ~= exact-MLE selection. Short and
    medium series keep the exact search, where it is cheap and more accurate.
    """
    return n_obs > _ARIMA_APPROX_MIN_OBS


def _tabular_pipeline(estimator: Any) -> Pipeline:
    """Wrap a tabular estimator so string/categorical features are encoded before fit.

    Object-dtype columns are ordinal-encoded (categories unseen at predict time map to
    -1); numeric columns pass through unchanged. This keeps the executor's fit/predict
    code untouched and makes the saved champion self-contained — it encodes raw
    categorical input at inference. Identifier columns are dropped upstream (in the
    executor) and never reach this pipeline.
    """
    pre = ColumnTransformer(
        transformers=[(
            "cat",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            make_column_selector(dtype_include="object"),
        )],
        remainder="passthrough",
    )
    return Pipeline([("pre", pre), ("model", estimator)])


def build_logistic_regression(params: dict[str, Any]) -> Pipeline:
    return _tabular_pipeline(LogisticRegression(**params))


def build_random_forest_classifier(params: dict[str, Any]) -> Pipeline:
    return _tabular_pipeline(RandomForestClassifier(**params))


def build_lightgbm_classifier(params: dict[str, Any]) -> Pipeline:
    from lightgbm import LGBMClassifier
    return _tabular_pipeline(LGBMClassifier(**{**params, "verbosity": -1}))


def build_xgboost_classifier(params: dict[str, Any]) -> Pipeline:
    from xgboost import XGBClassifier
    return _tabular_pipeline(XGBClassifier(**{**params, "verbosity": 0}))


def build_catboost_classifier(params: dict[str, Any]) -> Pipeline:
    from catboost import CatBoostClassifier
    return _tabular_pipeline(CatBoostClassifier(**{**params, "verbose": 0}))


def build_ridge(params: dict[str, Any]) -> Pipeline:
    return _tabular_pipeline(Ridge(**params))


def build_random_forest_regressor(params: dict[str, Any]) -> Pipeline:
    return _tabular_pipeline(RandomForestRegressor(**params))


def build_lightgbm_regressor(params: dict[str, Any]) -> Pipeline:
    from lightgbm import LGBMRegressor
    return _tabular_pipeline(LGBMRegressor(**{**params, "verbosity": -1}))


def build_xgboost_regressor(params: dict[str, Any]) -> Pipeline:
    from xgboost import XGBRegressor
    return _tabular_pipeline(XGBRegressor(**{**params, "verbosity": 0}))


def build_catboost_regressor(params: dict[str, Any]) -> Pipeline:
    from catboost import CatBoostRegressor
    return _tabular_pipeline(CatBoostRegressor(**{**params, "verbose": 0}))


def build_naive(spec: dict[str, Any]) -> Any:
    """Naive forecaster: predicts the last observed value."""
    from statsforecast import StatsForecast
    from statsforecast.models import Naive
    freq = spec["task_metadata"]["frequency"]
    return StatsForecast(models=[Naive()], freq=freq, n_jobs=1)


def build_seasonal_naive(spec: dict[str, Any]) -> Any:
    """Seasonal naive: predicts the value from one season ago."""
    from statsforecast import StatsForecast
    from statsforecast.models import SeasonalNaive
    freq = spec["task_metadata"]["frequency"]
    season_length = spec["params"].get("season_length", default_season_length(freq))
    return StatsForecast(models=[SeasonalNaive(season_length=season_length)], freq=freq, n_jobs=1)


def build_ets(spec: dict[str, Any]) -> Any:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS
    freq = spec["task_metadata"]["frequency"]
    season_length = spec["params"].get("season_length", default_season_length(freq))
    return StatsForecast(models=[AutoETS(season_length=season_length)], freq=freq, n_jobs=1)


def build_auto_arima(spec: dict[str, Any]) -> Any:
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA
    task_metadata = spec["task_metadata"]
    freq = task_metadata["frequency"]
    season_length = spec["params"].get("season_length", default_season_length(freq))
    # Length-gated CSS approximation for the order search; final order is full MLE.
    # series_length is injected by the executor; absent (ad-hoc callers) -> exact.
    approximation = arima_use_approximation(task_metadata.get("series_length", 0))
    return StatsForecast(
        models=[AutoARIMA(season_length=season_length, approximation=approximation)],
        freq=freq, n_jobs=1,
    )


def _split_lags_from_params(params: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    """Pull `lags` out of params; the rest goes to the regressor."""
    p = dict(params)
    lags = int(p.pop("lags", 12))
    return lags, p


def _wrap_with_skforecast(estimator: Any, lags: int) -> Any:
    from skforecast.recursive import ForecasterRecursiveMultiSeries
    return ForecasterRecursiveMultiSeries(estimator=estimator, lags=lags)


def build_random_forest_forecaster(spec: dict[str, Any]) -> Any:
    lags, p = _split_lags_from_params(spec["params"])
    return _wrap_with_skforecast(RandomForestRegressor(**p), lags)


def build_extra_trees_forecaster(spec: dict[str, Any]) -> Any:
    from sklearn.ensemble import ExtraTreesRegressor
    lags, p = _split_lags_from_params(spec["params"])
    return _wrap_with_skforecast(ExtraTreesRegressor(**p), lags)


def build_gbm_forecaster(spec: dict[str, Any]) -> Any:
    from sklearn.ensemble import GradientBoostingRegressor
    lags, p = _split_lags_from_params(spec["params"])
    return _wrap_with_skforecast(GradientBoostingRegressor(**p), lags)


def build_lightgbm_forecaster(spec: dict[str, Any]) -> Any:
    from lightgbm import LGBMRegressor
    lags, p = _split_lags_from_params(spec["params"])
    return _wrap_with_skforecast(LGBMRegressor(**{**p, "verbosity": -1}), lags)


def build_xgboost_forecaster(spec: dict[str, Any]) -> Any:
    from xgboost import XGBRegressor
    lags, p = _split_lags_from_params(spec["params"])
    return _wrap_with_skforecast(XGBRegressor(**p), lags)


def build_svr_forecaster(spec: dict[str, Any]) -> Any:
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
