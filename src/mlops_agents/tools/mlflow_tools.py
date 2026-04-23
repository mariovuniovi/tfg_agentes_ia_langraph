"""MLflow experiment tracking and model registry tools.

These tools are deterministic wrappers around the MLflow API.
The training and evaluation agents call these to log runs and
query results; the deployment agent uses them to register models.
"""

import json
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
from langchain_core.tools import tool
from mlflow.tracking import MlflowClient

from mlops_agents.config.constants import MLFLOW_REGISTERED_MODEL_NAME
from mlops_agents.config.settings import settings
from mlops_agents.utils.logging import get_logger

logger = get_logger(__name__)


def _get_client() -> MlflowClient:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    return MlflowClient()


@tool
def log_experiment(
    model_path: str,
    model_type: str,
    hyperparameters_json: str,
    metrics_json: str,
    run_name: str = "agent-run",
) -> str:
    """Log a trained model and its metrics to MLflow.

    Args:
        model_path: Path to the pickled model file.
        model_type: Name of the model type (e.g., 'random_forest').
        hyperparameters_json: JSON dict of hyperparameters.
        metrics_json: JSON dict of metrics (e.g., accuracy, f1_score).
        run_name: Optional name for the MLflow run.

    Returns:
        JSON with the MLflow run_id and artifact URI.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    params = json.loads(hyperparameters_json)
    raw_metrics = json.loads(metrics_json)
    # MLflow only accepts scalar floats — skip any nested dicts the agent may pass
    metrics = {k: float(v) for k, v in raw_metrics.items() if isinstance(v, (int, float))}

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_param("model_type", model_type)
        mlflow.sklearn.log_model(model, artifact_path="model")
        run_id = run.info.run_id

    logger.info(f"Logged experiment: run_id={run_id}")
    return json.dumps({"run_id": run_id, "model_uri": f"runs:/{run_id}/model"})


@tool
def get_best_run(metric: str = "accuracy", top_n: int = 5) -> str:
    """Query MLflow for the best run in the current experiment by a given metric.

    Args:
        metric: The metric name to rank by (default 'accuracy').
        top_n: Number of top runs to return.

    Returns:
        JSON list of top runs with run_id, metrics, and params.
    """
    client = _get_client()
    experiment = client.get_experiment_by_name(settings.mlflow_experiment_name)
    if experiment is None:
        return json.dumps({"error": f"Experiment '{settings.mlflow_experiment_name}' not found."})

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=top_n,
    )
    results = [
        {
            "run_id": r.info.run_id,
            "metrics": r.data.metrics,
            "params": r.data.params,
            "model_uri": f"runs:/{r.info.run_id}/model",
        }
        for r in runs
    ]
    return json.dumps(results)


@tool
def register_model(run_id: str, model_name: str = MLFLOW_REGISTERED_MODEL_NAME) -> str:
    """Register a trained model in the MLflow Model Registry.

    Args:
        run_id: The MLflow run ID whose model artifact to register.
        model_name: Name for the registered model (default from constants).

    Returns:
        JSON with registered model name and version.
    """
    client = _get_client()
    model_uri = f"runs:/{run_id}/model"
    mv = mlflow.register_model(model_uri, model_name)
    logger.info(f"Registered model '{model_name}' version {mv.version}")
    return json.dumps({"model_name": model_name, "version": mv.version, "run_id": run_id})


@tool
def set_model_alias(model_name: str, alias: str, version: int) -> str:
    """Set an alias (e.g., 'champion') on a registered model version.

    Args:
        model_name: Registered model name.
        alias: Alias to assign (e.g., 'champion', 'challenger', 'staging').
        version: Model version number to tag.

    Returns:
        JSON confirming the alias was set.
    """
    client = _get_client()
    client.set_registered_model_alias(model_name, alias, version)
    logger.info(f"Set alias '{alias}' → {model_name} v{version}")
    return json.dumps({"model_name": model_name, "alias": alias, "version": version})
