"""Unit tests for MLflow tools — all MLflow calls are mocked.

No real MLflow server or experiment tracking is needed for these tests.
The tests verify: correct return shapes, error handling, and that the
right MLflow API calls are made with the expected arguments.
"""

import json
import pickle
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_run(run_id: str, metrics: dict, params: dict) -> MagicMock:
    run = MagicMock()
    run.info.run_id = run_id
    run.data.metrics = metrics
    run.data.params = params
    return run


# ---------------------------------------------------------------------------
# get_best_run
# ---------------------------------------------------------------------------

@patch("mlops_agents.tools.mlflow_tools.MlflowClient")
@patch("mlops_agents.tools.mlflow_tools.mlflow")
def test_get_best_run_returns_error_when_experiment_not_found(mock_mlflow, mock_client_cls):
    mock_client = MagicMock()
    mock_client.get_experiment_by_name.return_value = None
    mock_client_cls.return_value = mock_client

    from mlops_agents.tools.mlflow_tools import get_best_run

    result = json.loads(get_best_run.invoke({"metric": "accuracy", "top_n": 5}))
    assert "error" in result


@patch("mlops_agents.tools.mlflow_tools.MlflowClient")
@patch("mlops_agents.tools.mlflow_tools.mlflow")
def test_get_best_run_returns_list_when_runs_exist(mock_mlflow, mock_client_cls):
    mock_client = MagicMock()
    mock_experiment = MagicMock()
    mock_experiment.experiment_id = "1"
    mock_client.get_experiment_by_name.return_value = mock_experiment

    runs = [
        _make_mock_run("run-abc", {"accuracy": 0.95}, {"n_estimators": "100"}),
        _make_mock_run("run-def", {"accuracy": 0.90}, {"n_estimators": "50"}),
    ]
    mock_client.search_runs.return_value = runs
    mock_client_cls.return_value = mock_client

    from mlops_agents.tools.mlflow_tools import get_best_run

    result = json.loads(get_best_run.invoke({"metric": "accuracy", "top_n": 2}))

    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0]["run_id"] == "run-abc"
    assert result[0]["metrics"]["accuracy"] == 0.95
    assert "model_uri" in result[0]
    assert result[0]["model_uri"] == "runs:/run-abc/model"


@patch("mlops_agents.tools.mlflow_tools.MlflowClient")
@patch("mlops_agents.tools.mlflow_tools.mlflow")
def test_get_best_run_orders_by_requested_metric(mock_mlflow, mock_client_cls):
    """Verifies that search_runs is called with the correct order_by clause."""
    mock_client = MagicMock()
    mock_experiment = MagicMock()
    mock_experiment.experiment_id = "1"
    mock_client.get_experiment_by_name.return_value = mock_experiment
    mock_client.search_runs.return_value = []
    mock_client_cls.return_value = mock_client

    from mlops_agents.tools.mlflow_tools import get_best_run

    get_best_run.invoke({"metric": "f1_score", "top_n": 3})

    mock_client.search_runs.assert_called_once_with(
        experiment_ids=["1"],
        order_by=["metrics.f1_score DESC"],
        max_results=3,
    )


# ---------------------------------------------------------------------------
# log_experiment
# ---------------------------------------------------------------------------

@patch("mlops_agents.tools.mlflow_tools.mlflow")
def test_log_experiment_returns_run_id_and_uri(mock_mlflow, tmp_path):
    """log_experiment should return a dict with run_id and model_uri."""
    # Create a real pickled model file for the tool to load
    from sklearn.dummy import DummyClassifier
    model = DummyClassifier()
    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    mock_run = MagicMock()
    mock_run.__enter__ = lambda s: s
    mock_run.__exit__ = MagicMock(return_value=False)
    mock_run.info.run_id = "test-run-123"
    mock_mlflow.start_run.return_value = mock_run

    from mlops_agents.tools.mlflow_tools import log_experiment

    result = json.loads(log_experiment.invoke({
        "model_path": str(model_path),
        "model_type": "random_forest",
        "hyperparameters_json": '{"n_estimators": 100}',
        "metrics_json": '{"accuracy": 0.92, "f1_score": 0.89}',
        "run_name": "test-run",
    }))

    assert "run_id" in result
    assert "model_uri" in result
    assert result["run_id"] == "test-run-123"
    assert result["model_uri"] == "runs:/test-run-123/model"


@patch("mlops_agents.tools.mlflow_tools.mlflow")
def test_log_experiment_calls_log_params_and_metrics(mock_mlflow, tmp_path):
    """Verify that log_params and log_metrics are called with correct arguments."""
    from sklearn.dummy import DummyClassifier
    model = DummyClassifier()
    model_path = tmp_path / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    mock_run = MagicMock()
    mock_run.__enter__ = lambda s: s
    mock_run.__exit__ = MagicMock(return_value=False)
    mock_run.info.run_id = "run-xyz"
    mock_mlflow.start_run.return_value = mock_run

    from mlops_agents.tools.mlflow_tools import log_experiment

    log_experiment.invoke({
        "model_path": str(model_path),
        "model_type": "gradient_boosting",
        "hyperparameters_json": '{"learning_rate": 0.1}',
        "metrics_json": '{"accuracy": 0.88}',
    })

    mock_mlflow.log_params.assert_called_once_with({"learning_rate": 0.1})
    mock_mlflow.log_metrics.assert_called_once_with({"accuracy": 0.88})


# ---------------------------------------------------------------------------
# register_model
# ---------------------------------------------------------------------------

@patch("mlops_agents.tools.mlflow_tools.MlflowClient")
@patch("mlops_agents.tools.mlflow_tools.mlflow")
def test_register_model_returns_name_and_version(mock_mlflow, mock_client_cls):
    mock_mv = MagicMock()
    mock_mv.version = 1
    mock_mlflow.register_model.return_value = mock_mv

    from mlops_agents.tools.mlflow_tools import register_model

    result = json.loads(register_model.invoke({
        "run_id": "abc123",
        "model_name": "test-model",
    }))

    assert result["model_name"] == "test-model"
    assert result["version"] == 1
    assert result["run_id"] == "abc123"


@patch("mlops_agents.tools.mlflow_tools.MlflowClient")
@patch("mlops_agents.tools.mlflow_tools.mlflow")
def test_register_model_calls_mlflow_register_with_correct_uri(mock_mlflow, mock_client_cls):
    mock_mv = MagicMock()
    mock_mv.version = 2
    mock_mlflow.register_model.return_value = mock_mv

    from mlops_agents.tools.mlflow_tools import register_model

    register_model.invoke({"run_id": "run-99", "model_name": "my-model"})

    mock_mlflow.register_model.assert_called_once_with("runs:/run-99/model", "my-model")


# ---------------------------------------------------------------------------
# set_model_alias
# ---------------------------------------------------------------------------

@patch("mlops_agents.tools.mlflow_tools.MlflowClient")
@patch("mlops_agents.tools.mlflow_tools.mlflow")
def test_set_model_alias_returns_confirmation(mock_mlflow, mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    from mlops_agents.tools.mlflow_tools import set_model_alias

    result = json.loads(set_model_alias.invoke({
        "model_name": "my-model",
        "alias": "champion",
        "version": 3,
    }))

    assert result["model_name"] == "my-model"
    assert result["alias"] == "champion"
    assert result["version"] == 3


@patch("mlops_agents.tools.mlflow_tools.MlflowClient")
@patch("mlops_agents.tools.mlflow_tools.mlflow")
def test_set_model_alias_calls_client_with_correct_args(mock_mlflow, mock_client_cls):
    mock_client = MagicMock()
    mock_client_cls.return_value = mock_client

    from mlops_agents.tools.mlflow_tools import set_model_alias

    set_model_alias.invoke({
        "model_name": "my-model",
        "alias": "challenger",
        "version": 2,
    })

    mock_client.set_registered_model_alias.assert_called_once_with(
        "my-model", "challenger", 2
    )
