"""TDD for the agentic cost/timing aggregator (scripts/agentic_cost_aggregate.py).

The aggregator reconstructs per-node compute time and cost from a pipeline
EventLog. The fixture below uses the REAL event timestamps from a completed
Bakery run (run-5ed31237) so the reconstruction is anchored to ground truth.

Key methodology assertions:
- per-node time = gap between consecutive node-completion boundaries
  (run_info start → routing(next=X) marks X's completion).
- HITL gate nodes (dataset_approval / deployment_approval) are EXCLUDED, which
  is exactly where the human pause lives — so pauses never inflate a worker node.
- cost = sum of token_usage.estimated_cost_usd grouped by node.
"""
from __future__ import annotations

import pytest

from scripts.agentic_cost_aggregate import (
    HITL_NODES,
    aggregate_run,
    reconstruct_node_costs,
    reconstruct_node_times,
    summarize_runs,
)


def _routing(ts: float, node: str) -> dict:
    return {"type": "routing", "agent": "controller", "timestamp_ms": ts, "data": {"next": node}}


def _tok(ts: float, node: str, cost: float) -> dict:
    return {
        "type": "token_usage",
        "agent": node,
        "timestamp_ms": ts,
        "data": {"node": node, "estimated_cost_usd": cost},
    }


# Real boundary timestamps from run-5ed31237 (completed forecasting Bakery run,
# evaluation_passed=false so no deployer). Human pause sits inside the
# dataset_approval segment (hitl_request 125428 → hitl_resolved 190187 ≈ 64.8s).
RUN1_EVENTS: list[dict] = [
    {"type": "run_info", "agent": "system", "timestamp_ms": 1782581107481.2776, "data": {}},
    # data_validator work + its token_usage events
    _tok(1782581110662.2124, "data_validator", 0.0035902499999999997),
    _tok(1782581114035.3772, "data_validator", 0.00419175),
    _tok(1782581115342.39, "data_validator", 0.0039675000000000005),
    _tok(1782581118491.9766, "data_validator", 0.0040545),
    _tok(1782581119888.176, "data_validator", 0.0040035),
    _tok(1782581121977.4426, "data_validator", 0.00400725),
    _tok(1782581125413.9863, "data_validator", 0.005242500000000001),
    _routing(1782581125422.9675, "data_validator"),
    {"type": "hitl_request", "agent": "data_validation", "timestamp_ms": 1782581125428.957, "data": {}},
    {"type": "hitl_resolved", "agent": "data_validation", "timestamp_ms": 1782581190187.9995, "data": {}},
    _routing(1782581190194.1406, "dataset_approval"),
    # planner work + token_usage
    _tok(1782581193434.7795, "planner", 0.00295875),
    _tok(1782581194421.6423, "planner", 0.0032775),
    _tok(1782581248344.5093, "planner", 0.04144425),
    _routing(1782581248366.6782, "planner"),
    {"type": "training_complete", "agent": "executor", "timestamp_ms": 1782581276246.2178, "data": {}},
    _routing(1782581276246.2295, "executor"),
    _routing(1782581276354.351, "evaluation"),
    _tok(1782581283068.7844, "report_writer", 0.0047522499999999995),
    _routing(1782581283103.1294, "report_writer"),
    {"type": "run_complete", "agent": "controller", "timestamp_ms": 1782581283105.5273, "data": {}},
]


def test_node_times_match_real_boundaries():
    times = reconstruct_node_times(RUN1_EVENTS)
    assert times["data_validator"] == pytest.approx(17.9417, abs=0.01)
    assert times["planner"] == pytest.approx(58.1725, abs=0.01)
    assert times["executor"] == pytest.approx(27.8796, abs=0.01)
    assert times["evaluation"] == pytest.approx(0.1081, abs=0.01)
    assert times["report_writer"] == pytest.approx(6.7488, abs=0.01)


def test_hitl_pause_is_excluded_not_attributed_to_a_worker():
    times = reconstruct_node_times(RUN1_EVENTS)
    # The ~64.8s human pause lived in the dataset_approval segment and must not
    # appear as a worker-node time nor leak into planner.
    for hitl in HITL_NODES:
        assert hitl not in times
    assert times["planner"] < 70.0  # would be ~123s if the pause leaked in


def test_node_costs_grouped_by_node():
    costs = reconstruct_node_costs(RUN1_EVENTS)
    assert costs["data_validator"] == pytest.approx(0.02905725, abs=1e-6)
    assert costs["planner"] == pytest.approx(0.04768050, abs=1e-6)
    assert costs["report_writer"] == pytest.approx(0.00475225, abs=1e-6)


def test_aggregate_run_combines_time_and_cost():
    agg = aggregate_run(RUN1_EVENTS)
    assert agg["planner"]["time_s"] == pytest.approx(58.1725, abs=0.01)
    assert agg["planner"]["cost_usd"] == pytest.approx(0.04768050, abs=1e-6)
    assert agg["total_compute_s"] == pytest.approx(
        17.9417 + 58.1725 + 27.8796 + 0.1081 + 6.7488, abs=0.05
    )
    assert agg["total_cost_usd"] == pytest.approx(
        0.02905725 + 0.04768050 + 0.00475225, abs=1e-6
    )


def test_summarize_runs_means_and_std():
    # Two runs with simple node times to check mean ± std plumbing.
    r1 = {"planner": {"time_s": 60.0, "cost_usd": 0.05}}
    r2 = {"planner": {"time_s": 50.0, "cost_usd": 0.05}}
    summary = summarize_runs([r1, r2])
    assert summary["planner"]["time_s_mean"] == pytest.approx(55.0)
    assert summary["planner"]["time_s_std"] == pytest.approx(5.0, abs=0.01)
    assert summary["planner"]["cost_usd_mean"] == pytest.approx(0.05)
