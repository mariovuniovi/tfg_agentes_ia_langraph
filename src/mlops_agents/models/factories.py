"""Factory functions for every registered model.

Each factory takes a hyperparameter dict (and for forecasting, task_metadata)
and returns a sklearn-compatible estimator or forecaster. Factories are referenced
by string name from registry.yaml.
"""

from __future__ import annotations

from typing import Any, Callable

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


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


FACTORY_REGISTRY: dict[str, Callable[..., Any]] = {
    "build_logistic_regression":      build_logistic_regression,
    "build_random_forest_classifier": build_random_forest_classifier,
    "build_lightgbm_classifier":      build_lightgbm_classifier,
    "build_xgboost_classifier":       build_xgboost_classifier,
    "build_catboost_classifier":      build_catboost_classifier,
}
