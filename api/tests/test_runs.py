"""Tests for /runs endpoints."""
from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

import api.services.run_store as run_store_module
from api.main import app


@pytest.fixture()
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_post_runs_returns_run_id(client):
    with patch("api.routers.runs.pipeline_task"):
        resp = await client.post("/runs", json={"dataset_paths": ["data/samples/iris_measurements.csv"]})
    assert resp.status_code == 200
    body = resp.json()
    assert "run_id" in body
    assert isinstance(body["run_id"], str)


@pytest.mark.asyncio
async def test_get_run_status_running(client):
    with patch("api.routers.runs.pipeline_task"):
        start = await client.post("/runs", json={"dataset_paths": ["data/samples/iris_measurements.csv"]})
    run_id = start.json()["run_id"]
    resp = await client.get(f"/runs/{run_id}")
    assert resp.status_code == 200
    assert resp.json()["status"] == "running"


@pytest.mark.asyncio
async def test_get_run_status_unknown(client):
    resp = await client.get("/runs/does-not-exist")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_approve_run_not_awaiting_returns_400(client):
    with patch("api.routers.runs.pipeline_task"):
        start = await client.post("/runs", json={"dataset_paths": ["data/samples/iris_measurements.csv"]})
    run_id = start.json()["run_id"]
    resp = await client.post(f"/runs/{run_id}/approve", json={"decision": "approve"})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_approve_run_awaiting_returns_ok(client):
    with patch("api.routers.runs.pipeline_task"):
        start = await client.post("/runs", json={"dataset_paths": ["data/samples/iris_measurements.csv"]})
    run_id = start.json()["run_id"]
    entry = run_store_module.get_entry(run_id)
    entry.status = "awaiting_approval"
    resp = await client.post(f"/runs/{run_id}/approve", json={"decision": "approve"})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert entry.hitl_decision == "approve"
    assert entry.hitl_event.is_set()


@pytest.mark.asyncio
async def test_get_run_events_empty(client):
    with patch("api.routers.runs.pipeline_task"):
        start = await client.post("/runs", json={"dataset_paths": ["data/samples/iris_measurements.csv"]})
    run_id = start.json()["run_id"]
    resp = await client.get(f"/runs/{run_id}/events")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_approve_saves_comment(client):
    with patch("api.routers.runs.pipeline_task"):
        start = await client.post("/runs", json={"dataset_paths": ["data/samples/iris_measurements.csv"]})
    run_id = start.json()["run_id"]
    entry = run_store_module.get_entry(run_id)
    entry.status = "awaiting_approval"
    resp = await client.post(f"/runs/{run_id}/approve", json={"decision": "reject", "comment": "rename column X"})
    assert resp.status_code == 200
    assert entry.hitl_comment == "rename column X"
