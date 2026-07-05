import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.services import run_store


@pytest.fixture
def client(tmp_path):
    csv = tmp_path / "p.csv"
    pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "y", "z", "w", "v"]}).to_csv(csv, index=False)
    run_store._store.clear()
    entry = run_store.create_entry("run-1", graph_config={})
    entry.processed_dataset_path = str(csv)
    yield TestClient(app)
    run_store._store.clear()


def test_dataset_preview_pagination(client):
    r = client.get("/runs/run-1/dataset-preview?limit=2&offset=0")
    assert r.status_code == 200
    body = r.json()
    assert body["total_rows"] == 5
    assert len(body["rows"]) == 2
    assert body["rows"][0]["a"] == 1
    cols = {c["name"] for c in body["columns"]}
    assert cols == {"a", "b"}


def test_dataset_preview_404_unknown_run(client):
    assert client.get("/runs/unknown/dataset-preview").status_code == 404


def test_dataset_download_streams_csv(client):
    r = client.get("/runs/run-1/dataset-download")
    assert r.status_code == 200
    assert "text/csv" in r.headers["content-type"]
    assert b"a,b" in r.content
