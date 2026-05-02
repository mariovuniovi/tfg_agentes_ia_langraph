"""Tests for MlflowService."""
from datetime import datetime
from unittest.mock import MagicMock, patch
import pytest
from api.services.mlflow_client import MlflowService
from api.models.experiment import ExperimentOut, RunOut


def _mock_experiment(exp_id="1", name="mlops-agents"):
    exp = MagicMock()
    exp.experiment_id = exp_id
    exp.name = name
    return exp


def _mock_run(run_id="abc123", status="FINISHED", start_time_ms=1_700_000_000_000):
    run = MagicMock()
    run.info.run_id = run_id
    run.info.run_name = "run-1"
    run.info.status = status
    run.info.start_time = start_time_ms
    run.data.params = {"model_type": "RandomForest"}
    run.data.metrics = {"accuracy": 0.94, "f1": 0.91}
    return run


def _mock_metric_history(values: list[float]):
    return [MagicMock(step=i, value=v) for i, v in enumerate(values)]


def test_list_experiments():
    with patch("api.services.mlflow_client.MlflowClient") as MockClient:
        MockClient.return_value.search_experiments.return_value = [
            _mock_experiment("1", "mlops-agents"),
            _mock_experiment("2", "test-exp"),
        ]
        svc = MlflowService()
        exps = svc.list_experiments()

    assert len(exps) == 2
    assert all(isinstance(e, ExperimentOut) for e in exps)
    assert exps[0].name == "mlops-agents"


def test_get_runs_returns_run_out_list():
    with patch("api.services.mlflow_client.MlflowClient") as MockClient:
        client = MockClient.return_value
        client.search_runs.return_value = [_mock_run()]
        client.get_metric_history.side_effect = lambda run_id, metric: _mock_metric_history([0.8, 0.9])

        svc = MlflowService()
        runs = svc.get_runs("1")

    assert len(runs) == 1
    run = runs[0]
    assert isinstance(run, RunOut)
    assert run.run_id == "abc123"
    assert run.metrics["accuracy"] == 0.94
    assert len(run.metric_series) == 2  # accuracy + f1
    # line_style cycles: solid, dashed
    styles = [ms.line_style for ms in run.metric_series]
    assert styles == ["solid", "dashed"]


def test_get_runs_empty():
    with patch("api.services.mlflow_client.MlflowClient") as MockClient:
        MockClient.return_value.search_runs.return_value = []
        svc = MlflowService()
        runs = svc.get_runs("1")
    assert runs == []


# ── Router-level tests ─────────────────────────────────────────────────────────
from datetime import datetime, timezone
from httpx import AsyncClient, ASGITransport
from api.main import app


@pytest.mark.asyncio
async def test_get_experiments_endpoint():
    with patch("api.routers.experiments.MlflowService") as MockSvc:
        MockSvc.return_value.list_experiments.return_value = [
            ExperimentOut(experiment_id="1", name="mlops-agents"),
        ]
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/experiments")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["name"] == "mlops-agents"


@pytest.mark.asyncio
async def test_get_experiment_runs_endpoint():
    from api.models.experiment import MetricSeries
    with patch("api.routers.experiments.MlflowService") as MockSvc:
        MockSvc.return_value.get_runs.return_value = [
            RunOut(
                run_id="abc",
                run_name="run-1",
                status="FINISHED",
                start_time=datetime.now(timezone.utc),
                params={},
                metrics={"accuracy": 0.94},
                metric_series=[
                    MetricSeries(name="accuracy", steps=[1], values=[0.94], line_style="solid")
                ],
            )
        ]
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.get("/experiments/1/runs")
    assert resp.status_code == 200
    runs = resp.json()
    assert runs[0]["run_id"] == "abc"
    assert runs[0]["metric_series"][0]["line_style"] == "solid"
