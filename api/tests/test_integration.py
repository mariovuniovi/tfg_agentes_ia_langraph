"""Integration tests — hit the real graph and MLflow. Require running services.

Run with: uv run pytest api/tests/test_integration.py -m integration -v
"""
import asyncio
import pytest
from httpx import AsyncClient, ASGITransport
from api.main import app


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_health_endpoint_with_real_services():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["graph"] is True


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_full_pipeline_run_streams_events():
    """Start a real pipeline run and consume WebSocket events until run_complete."""
    import websockets
    import json

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post(
            "/runs",
            json={"dataset_paths": ["data/samples/iris_measurements.csv",
                                    "data/samples/iris_labels.csv"]},
        )
    assert resp.status_code == 200
    run_id = resp.json()["run_id"]

    # Give the background task a moment to start
    await asyncio.sleep(0.5)

    received_types: list[str] = []
    async with websockets.connect(f"ws://localhost:8000/ws/{run_id}") as ws:
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=120)
            event = json.loads(raw)
            received_types.append(event["type"])
            if event["type"] == "run_complete":
                break

    assert "routing" in received_types
    assert "run_complete" in received_types
