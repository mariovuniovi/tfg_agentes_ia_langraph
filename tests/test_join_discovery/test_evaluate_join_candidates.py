from __future__ import annotations

import json
import pytest
from mlops_agents.tools.join_discovery_tools import evaluate_join_candidates


@pytest.fixture
def perfect_overlap_csvs(tmp_path):
    e = tmp_path / "energy.csv"
    w = tmp_path / "weather.csv"
    e.write_text("week_date,kwh_consumed\n2024-01-01,100\n2024-01-08,120\n")
    w.write_text("week_date,avg_temp_c\n2024-01-01,10\n2024-01-08,12\n")
    return {
        "paths": {"energy": str(e), "weather": str(w)},
        "candidate": {
            "candidate_id": "join_001",
            "left_dataset": "energy",
            "left_column": "week_date",
            "right_dataset": "weather",
            "right_column": "week_date",
        }
    }


def test_perfect_overlap_metrics(perfect_overlap_csvs) -> None:
    result = json.loads(evaluate_join_candidates.invoke({
        "candidates_json": json.dumps([perfect_overlap_csvs["candidate"]]),
        "raw_paths_json": json.dumps(perfect_overlap_csvs["paths"]),
    }))
    assert not result["errors"]
    ev = result["evaluations"][0]
    assert ev["left_coverage"] == 1.0
    assert ev["right_coverage"] == 1.0
    assert ev["jaccard"] == 1.0
    assert ev["row_multiplier_left"] == pytest.approx(1.0)
    assert ev["join_explosion_risk"] == "low"


def test_subset_coverage(tmp_path) -> None:
    base = tmp_path / "base.csv"
    enrichment = tmp_path / "enrichment.csv"
    base.write_text("id\n1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n")
    enrichment.write_text("id\n1\n2\n3\n4\n5\n6\n7\n8\n")
    result = json.loads(evaluate_join_candidates.invoke({
        "candidates_json": json.dumps([{
            "candidate_id": "join_001",
            "left_dataset": "base",
            "left_column": "id",
            "right_dataset": "enrichment",
            "right_column": "id",
        }]),
        "raw_paths_json": json.dumps({"base": str(base), "enrichment": str(enrichment)}),
    }))
    ev = result["evaluations"][0]
    assert ev["left_coverage"] == pytest.approx(0.8)
    assert ev["right_coverage"] == 1.0
    assert ev["jaccard"] == pytest.approx(0.8)
    assert ev["containment"] == 1.0  # max(0.8, 1.0)


def test_many_to_many_detection(tmp_path) -> None:
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    left.write_text("key\nA\nA\nA\nB\n")
    right.write_text("key\nA\nA\nA\nA\nB\n")
    result = json.loads(evaluate_join_candidates.invoke({
        "candidates_json": json.dumps([{
            "candidate_id": "join_001",
            "left_dataset": "left",
            "left_column": "key",
            "right_dataset": "right",
            "right_column": "key",
        }]),
        "raw_paths_json": json.dumps({"left": str(left), "right": str(right)}),
    }))
    ev = result["evaluations"][0]
    assert ev["inferred_relationship"] == "many_to_many"
    assert any("many-to-many" in w for w in ev["warnings"])
