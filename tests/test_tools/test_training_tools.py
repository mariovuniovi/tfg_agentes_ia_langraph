"""Unit tests for training tools — uses real sklearn, no LLM calls.

tune_hyperparameters runs with n_trials=3 to stay fast (<2s per test).
train_model uses a monkeypatched MODELS_DIR so no files leak into the project root.
"""

import json
import pickle
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RF_PARAMS = json.dumps({"n_estimators": 10})
GB_PARAMS = json.dumps({"n_estimators": 10, "learning_rate": 0.1})
LR_PARAMS = json.dumps({"C": 1.0, "max_iter": 200})


# ---------------------------------------------------------------------------
# tune_hyperparameters
# ---------------------------------------------------------------------------

def test_tune_hyperparameters_random_forest_returns_best_params(larger_csv):
    from mlops_agents.tools.training_tools import tune_hyperparameters

    result = json.loads(tune_hyperparameters.invoke({
        "dataset_path": str(larger_csv),
        "model_type": "random_forest",
        "n_trials": 3,
    }))

    assert result["model_type"] == "random_forest"
    assert "best_params" in result
    assert "n_estimators" in result["best_params"]
    assert "best_cv_f1" in result
    assert 0.0 <= result["best_cv_f1"] <= 1.0
    assert result["n_trials"] == 3


def test_tune_hyperparameters_logistic_regression_returns_best_params(larger_csv):
    from mlops_agents.tools.training_tools import tune_hyperparameters

    result = json.loads(tune_hyperparameters.invoke({
        "dataset_path": str(larger_csv),
        "model_type": "logistic_regression",
        "n_trials": 3,
    }))

    assert result["model_type"] == "logistic_regression"
    assert "C" in result["best_params"]


def test_tune_hyperparameters_gradient_boosting_returns_best_params(larger_csv):
    from mlops_agents.tools.training_tools import tune_hyperparameters

    result = json.loads(tune_hyperparameters.invoke({
        "dataset_path": str(larger_csv),
        "model_type": "gradient_boosting",
        "n_trials": 3,
    }))

    assert result["model_type"] == "gradient_boosting"
    assert "learning_rate" in result["best_params"]


# ---------------------------------------------------------------------------
# train_model
# ---------------------------------------------------------------------------

def test_train_model_random_forest_returns_accuracy(larger_csv, tmp_path, monkeypatch):
    import mlops_agents.tools.training_tools as tt
    monkeypatch.setattr(tt, "MODELS_DIR", tmp_path)

    from mlops_agents.tools.training_tools import train_model

    result = json.loads(train_model.invoke({
        "dataset_path": str(larger_csv),
        "model_type": "random_forest",
        "hyperparameters_json": RF_PARAMS,
    }))

    assert result["model_type"] == "random_forest"
    assert 0.0 <= result["train_accuracy"] <= 1.0
    assert 0.0 <= result["val_accuracy"] <= 1.0
    assert "classification_report" in result


def test_train_model_gradient_boosting_returns_accuracy(larger_csv, tmp_path, monkeypatch):
    import mlops_agents.tools.training_tools as tt
    monkeypatch.setattr(tt, "MODELS_DIR", tmp_path)

    from mlops_agents.tools.training_tools import train_model

    result = json.loads(train_model.invoke({
        "dataset_path": str(larger_csv),
        "model_type": "gradient_boosting",
        "hyperparameters_json": GB_PARAMS,
    }))

    assert result["model_type"] == "gradient_boosting"
    assert 0.0 <= result["val_accuracy"] <= 1.0


def test_train_model_logistic_regression_returns_accuracy(larger_csv, tmp_path, monkeypatch):
    import mlops_agents.tools.training_tools as tt
    monkeypatch.setattr(tt, "MODELS_DIR", tmp_path)

    from mlops_agents.tools.training_tools import train_model

    result = json.loads(train_model.invoke({
        "dataset_path": str(larger_csv),
        "model_type": "logistic_regression",
        "hyperparameters_json": LR_PARAMS,
    }))

    assert result["model_type"] == "logistic_regression"
    assert 0.0 <= result["val_accuracy"] <= 1.0


def test_train_model_unknown_type_returns_error(larger_csv, tmp_path, monkeypatch):
    import mlops_agents.tools.training_tools as tt
    monkeypatch.setattr(tt, "MODELS_DIR", tmp_path)

    from mlops_agents.tools.training_tools import train_model

    result = json.loads(train_model.invoke({
        "dataset_path": str(larger_csv),
        "model_type": "neural_network",
        "hyperparameters_json": "{}",
    }))

    assert "error" in result
    assert "neural_network" in result["error"]


def test_train_model_saves_pkl_to_models_dir(larger_csv, tmp_path, monkeypatch):
    import mlops_agents.tools.training_tools as tt
    monkeypatch.setattr(tt, "MODELS_DIR", tmp_path)

    from mlops_agents.tools.training_tools import train_model

    result = json.loads(train_model.invoke({
        "dataset_path": str(larger_csv),
        "model_type": "random_forest",
        "hyperparameters_json": RF_PARAMS,
    }))

    model_path = Path(result["model_path"])
    assert model_path.exists()
    assert model_path.suffix == ".pkl"


def test_train_model_saved_pkl_is_loadable(larger_csv, tmp_path, monkeypatch):
    """The saved .pkl should deserialize back to a fitted sklearn estimator."""
    import mlops_agents.tools.training_tools as tt
    monkeypatch.setattr(tt, "MODELS_DIR", tmp_path)

    from mlops_agents.tools.training_tools import train_model

    result = json.loads(train_model.invoke({
        "dataset_path": str(larger_csv),
        "model_type": "random_forest",
        "hyperparameters_json": RF_PARAMS,
    }))

    with open(result["model_path"], "rb") as f:
        model = pickle.load(f)

    assert hasattr(model, "predict")
    assert hasattr(model, "score")


def test_train_model_hyperparameters_echoed_in_result(larger_csv, tmp_path, monkeypatch):
    import mlops_agents.tools.training_tools as tt
    monkeypatch.setattr(tt, "MODELS_DIR", tmp_path)

    from mlops_agents.tools.training_tools import train_model

    result = json.loads(train_model.invoke({
        "dataset_path": str(larger_csv),
        "model_type": "random_forest",
        "hyperparameters_json": RF_PARAMS,
    }))

    assert result["hyperparameters"]["n_estimators"] == 10
