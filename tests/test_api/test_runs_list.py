from fastapi.testclient import TestClient

from api.main import app
from api.services import run_store


def test_runs_list_returns_recent(monkeypatch):
    run_store._store.clear()
    e1 = run_store.create_entry("a", {})
    e1.status = "complete"
    e2 = run_store.create_entry("b", {})
    e2.status = "running"
    r = TestClient(app).get("/runs?limit=10")
    assert r.status_code == 200
    body = r.json()
    ids = [x["run_id"] for x in body]
    assert set(ids) == {"a", "b"}
    run_store._store.clear()
