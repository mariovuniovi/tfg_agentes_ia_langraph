from __future__ import annotations

import json
import pytest
import pandas as pd
from mlops_agents.tools.join_discovery_tools import execute_join_plan


EVALUATION = {
    "candidate_id": "join_001",
    "left_dataset": "energy", "left_column": "week_date",
    "right_dataset": "weather", "right_column": "week_date",
    "left_distinct": 2, "right_distinct": 2, "intersection_count": 2,
    "left_coverage": 1.0, "right_coverage": 1.0, "jaccard": 1.0, "containment": 1.0,
    "left_unique_ratio": 1.0, "right_unique_ratio": 1.0,
    "inferred_relationship": "one_to_one",
    "estimated_inner_rows": 2, "estimated_left_rows": 2,
    "row_multiplier_left": 1.0, "join_explosion_risk": "low",
    "warnings": [],
}

SELECTIONS = {
    "base_dataset": {
        "dataset_name": "energy", "confidence": "high",
        "covered_target_columns": ["week_date", "kwh_consumed"],
        "missing_target_columns": ["avg_temp_c"],
        "reason": "contains target column", "warnings": [],
    },
    "selected": [{"candidate_id": "join_001", "columns_to_add": ["avg_temp_c"],
                  "confidence_after_evaluation": "high", "reason": "perfect overlap", "warnings": []}],
    "rejected": [],
    "unresolved_ambiguities": [],
    "warnings": [],
}


@pytest.fixture
def energy_csvs(tmp_path):
    energy = tmp_path / "energy.csv"
    weather = tmp_path / "weather.csv"
    energy.write_text("week_date,kwh_consumed\n2024-01-01,100\n2024-01-08,120\n")
    weather.write_text("week_date,avg_temp_c\n2024-01-01,10\n2024-01-08,12\n")
    return {"energy": str(energy), "weather": str(weather)}, str(tmp_path / "merged.csv")


def test_execute_join_plan_produces_merged_csv(energy_csvs) -> None:
    paths, output = energy_csvs
    result = json.loads(execute_join_plan.invoke({
        "selections_json": json.dumps(SELECTIONS),
        "evaluations_json": json.dumps({"evaluations": [EVALUATION], "errors": []}),
        "raw_paths_json": json.dumps(paths),
        "output_path": output,
    }))
    assert result["success"]
    df = pd.read_csv(output)
    assert "avg_temp_c" in df.columns
    assert len(df) == 2  # base rows preserved


def test_execute_join_plan_blocks_zero_coverage(energy_csvs) -> None:
    paths, output = energy_csvs
    zero_eval = {**EVALUATION, "left_coverage": 0.0, "intersection_count": 0}
    result = json.loads(execute_join_plan.invoke({
        "selections_json": json.dumps(SELECTIONS),
        "evaluations_json": json.dumps({"evaluations": [zero_eval], "errors": []}),
        "raw_paths_json": json.dumps(paths),
        "output_path": output,
    }))
    assert "error" in result
    assert "zero overlap" in result["error"]


def test_execute_join_plan_blocks_unevaluated_candidate(energy_csvs) -> None:
    paths, output = energy_csvs
    result = json.loads(execute_join_plan.invoke({
        "selections_json": json.dumps(SELECTIONS),
        "evaluations_json": json.dumps({"evaluations": [], "errors": []}),  # no evaluations
        "raw_paths_json": json.dumps(paths),
        "output_path": output,
    }))
    assert "error" in result
    assert "not evaluated" in result["error"]
