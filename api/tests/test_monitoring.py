"""Tests for /monitoring endpoints."""
import io
from unittest.mock import patch
import pytest
from httpx import AsyncClient, ASGITransport
from api.main import app


@pytest.fixture()
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_get_monitoring_latest_no_run(client):
    import api.services.run_store as rs
    rs._latest_drift_report = None
    resp = await client.get("/monitoring/latest")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_monitoring_latest_returns_report(client):
    import api.services.run_store as rs
    from datetime import datetime, timezone
    rs._latest_drift_report = {
        "dataset_drift": False,
        "drift_share": 0.0,
        "columns": [],
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    resp = await client.get("/monitoring/latest")
    assert resp.status_code == 200
    assert resp.json()["dataset_drift"] is False


@pytest.mark.asyncio
async def test_post_monitoring_drift_returns_report(client):
    csv_content = b"feature_1,feature_2,target\n1.0,0.5,0\n2.0,1.5,1\n"
    mock_report = {
        "dataset_drift": False,
        "drift_share": 0.0,
        "columns": [],
        "generated_at": "2026-04-23T00:00:00+00:00",
    }
    with patch("api.routers.monitoring.run_evidently", return_value=mock_report):
        resp = await client.post(
            "/monitoring/drift",
            data={"reference_index": "0", "current_index": "1"},
            files=[
                ("files", ("ref.csv", io.BytesIO(csv_content), "text/csv")),
                ("files", ("cur.csv", io.BytesIO(csv_content), "text/csv")),
            ],
        )
    assert resp.status_code == 200
    assert "dataset_drift" in resp.json()


@pytest.mark.asyncio
async def test_post_monitoring_drift_bad_index(client):
    csv_content = b"feature_1,target\n1.0,0\n"
    resp = await client.post(
        "/monitoring/drift",
        data={"reference_index": "0", "current_index": "5"},
        files=[("files", ("ref.csv", io.BytesIO(csv_content), "text/csv"))],
    )
    assert resp.status_code == 400
