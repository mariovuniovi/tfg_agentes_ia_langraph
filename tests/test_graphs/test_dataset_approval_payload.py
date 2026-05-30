import pandas as pd


def _capture(monkeypatch):
    from mlops_agents.graphs import approval_nodes
    captured: dict = {}
    def fake_interrupt(payload):
        captured["payload"] = payload
        return {"approved": True, "comment": ""}
    monkeypatch.setattr(approval_nodes, "interrupt", fake_interrupt)
    return captured


def test_payload_uses_dataset_preview_key_with_correct_shape(tmp_path, monkeypatch):
    csv = tmp_path / "x.csv"
    pd.DataFrame({"a": list(range(10)), "b": list("abcdefghij")}).to_csv(csv, index=False)
    captured = _capture(monkeypatch)
    from mlops_agents.graphs.approval_nodes import dataset_approval_node
    dataset_approval_node({
        "problem_type": "classification",
        "processed_dataset_path": str(csv),
        "validation_report": {"passed": True},
        "agent_attempt_counts": {},
    })
    payload = captured["payload"]
    assert "dataset_preview" in payload
    assert "preview" not in payload  # legacy key must be gone
    p = payload["dataset_preview"]
    assert p["path"] == str(csv)
    assert p["row_count"] == 10
    assert p["column_count"] == 2
    assert p["shape"] == [10, 2]
    assert {c["name"] for c in p["columns"]} == {"a", "b"}
    assert all("dtype" in c for c in p["columns"])
    assert len(p["sample_rows"]) == 5
    assert p["tail"] == []  # non-forecasting


def test_tail_populated_for_forecasting(tmp_path, monkeypatch):
    csv = tmp_path / "ts.csv"
    pd.DataFrame({"ds": list(range(10)), "y": list(range(10))}).to_csv(csv, index=False)
    captured = _capture(monkeypatch)
    from mlops_agents.graphs.approval_nodes import dataset_approval_node
    dataset_approval_node({
        "problem_type": "forecasting",
        "processed_dataset_path": str(csv),
        "validation_report": {"passed": True},
        "agent_attempt_counts": {},
    })
    p = captured["payload"]["dataset_preview"]
    assert len(p["tail"]) == 5
    assert p["tail"][-1] == {"ds": 9, "y": 9}
