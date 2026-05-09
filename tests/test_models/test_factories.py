"""Unit tests for model factories."""

import numpy as np
import pytest

from mlops_agents.models.factories import FACTORY_REGISTRY


@pytest.fixture
def tabular_classification_xy():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(60, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
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
