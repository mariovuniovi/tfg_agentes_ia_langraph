"""Tests for POST /uploads endpoint."""
import io
import pytest
from httpx import AsyncClient, ASGITransport
from api.main import app


@pytest.fixture()
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest.mark.asyncio
async def test_upload_csv_returns_paths(client, tmp_path, monkeypatch):
    monkeypatch.setattr("api.routers.uploads.UPLOAD_DIR", str(tmp_path))
    csv_bytes = b"col1,col2\n1,2\n3,4\n"
    resp = await client.post(
        "/uploads",
        files=[
            ("files", ("train.csv", io.BytesIO(csv_bytes), "text/csv")),
            ("files", ("test.csv",  io.BytesIO(csv_bytes), "text/csv")),
        ],
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "paths" in data
    assert len(data["paths"]) == 2
    assert all(p.endswith(".csv") for p in data["paths"])


@pytest.mark.asyncio
async def test_upload_no_files_returns_400(client):
    resp = await client.post("/uploads", files=[])
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_upload_non_csv_returns_422(client, tmp_path, monkeypatch):
    monkeypatch.setattr("api.routers.uploads.UPLOAD_DIR", str(tmp_path))
    resp = await client.post(
        "/uploads",
        files=[("files", ("data.txt", io.BytesIO(b"hello"), "text/plain"))],
    )
    assert resp.status_code == 422
