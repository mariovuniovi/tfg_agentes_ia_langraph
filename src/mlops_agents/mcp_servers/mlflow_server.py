"""MLflow MCP server — exposes MLflow tracking and registry as MCP tools.

Run via:
    uv run python -m mlops_agents.mcp_servers.mlflow_server

Configured in .claude/.mcp.json for use within Claude Code sessions.
"""

import json

import mlflow
from mcp.server.fastmcp import FastMCP

from mlops_agents.config.settings import settings

mcp = FastMCP("mlflow-server")


@mcp.tool()
def list_experiments() -> str:
    """List all MLflow experiments."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments()
    return json.dumps([
        {"id": e.experiment_id, "name": e.name, "lifecycle_stage": e.lifecycle_stage}
        for e in experiments
    ])


@mcp.tool()
def get_experiment_runs(experiment_name: str, max_results: int = 10) -> str:
    """Get recent runs for an MLflow experiment, ordered by start time."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        return json.dumps({"error": f"Experiment '{experiment_name}' not found."})
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=max_results,
    )
    return json.dumps([
        {"run_id": r.info.run_id, "metrics": r.data.metrics, "params": r.data.params}
        for r in runs
    ])


@mcp.tool()
def list_registered_models() -> str:
    """List all models in the MLflow Model Registry."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = mlflow.tracking.MlflowClient()
    models = client.search_registered_models()
    return json.dumps([
        {"name": m.name, "latest_versions": [v.version for v in m.latest_versions]}
        for m in models
    ])


if __name__ == "__main__":
    mcp.run(transport="stdio")
