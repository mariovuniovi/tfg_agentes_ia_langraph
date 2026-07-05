"""Reconstruct per-node compute time and cost from a pipeline EventLog.

This is the *honest* methodology behind the Chapter 6 cost table: it derives
each node's wall time purely from the EventLog's node-completion boundaries and
isolates the HITL human pause into the gate-node segments, which are then
dropped. Cost is summed from the token-usage events the app already emits.

Pure functions only (no langgraph / network imports) so they are unit-testable
offline and reusable by both the headless measurement harness and any
post-hoc analysis of saved run JSONs.

Boundary model
--------------
The pipeline streams a ``routing`` event ``{"next": X}`` the moment node ``X``
finishes (see ``api/services/pipeline.py``). So the ordered completion
timestamps are::

    run_info.ts , routing(next=A).ts , routing(next=B).ts , ... , run_complete.ts

Node ``X``'s time is ``completion(X) - completion(previous)``; the first node's
"previous" is ``run_info`` (graph start). HITL gate nodes
(``dataset_approval`` / ``deployment_approval``) own the segment that contains
the human pause, and are excluded from the reported worker rows.
"""
from __future__ import annotations

import statistics
from typing import Any

# Worker (reported) nodes, in pipeline order. deployer only appears when a
# champion is promoted (evaluation passed + gate-2 approved).
WORKER_NODES: tuple[str, ...] = (
    "data_validator",
    "planner",
    "executor",
    "evaluation",
    "report_writer",
    "deployer",
)

# HITL gate nodes — their segment holds the human pause, so they are dropped.
HITL_NODES: frozenset[str] = frozenset({"dataset_approval", "deployment_approval"})

# Deterministic (non-LLM) nodes — used only for the "LLM share" rollup.
DETERMINISTIC_NODES: frozenset[str] = frozenset({"executor", "evaluation", "deployer"})


def reconstruct_node_times(events: list[dict[str, Any]]) -> dict[str, float]:
    """Per-node compute time in seconds, HITL gate segments excluded.

    A node appearing twice (e.g. a data_validator auto-fix retry) accumulates.
    """
    start_ts: float | None = None
    for ev in events:
        if ev.get("type") == "run_info":
            start_ts = float(ev["timestamp_ms"])
            break
    if start_ts is None:
        return {}

    boundaries: list[tuple[float, str]] = []
    for ev in events:
        if ev.get("type") == "routing":
            nxt = (ev.get("data") or {}).get("next")
            if nxt:
                boundaries.append((float(ev["timestamp_ms"]), str(nxt)))

    times: dict[str, float] = {}
    prev_ts = start_ts
    for ts, node in boundaries:
        dur_s = (ts - prev_ts) / 1000.0
        if node not in HITL_NODES:
            times[node] = times.get(node, 0.0) + dur_s
        prev_ts = ts
    return times


def reconstruct_node_costs(events: list[dict[str, Any]]) -> dict[str, float]:
    """Per-node USD cost = sum of token_usage.estimated_cost_usd by node."""
    costs: dict[str, float] = {}
    for ev in events:
        if ev.get("type") != "token_usage":
            continue
        data = ev.get("data") or {}
        node = data.get("node") or ev.get("agent") or "unknown"
        cost = data.get("estimated_cost_usd") or 0.0
        costs[node] = costs.get(node, 0.0) + float(cost)
    return costs


def aggregate_run(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine time + cost per node for a single run, plus run totals."""
    times = reconstruct_node_times(events)
    costs = reconstruct_node_costs(events)
    nodes = sorted(set(times) | set(costs), key=lambda n: (n not in WORKER_NODES, n))

    out: dict[str, Any] = {}
    total_time = 0.0
    total_cost = 0.0
    for node in nodes:
        t = times.get(node, 0.0)
        c = costs.get(node, 0.0)
        out[node] = {"time_s": t, "cost_usd": c}
        total_time += t
        total_cost += c

    llm_time = sum(
        v["time_s"] for n, v in out.items() if n not in DETERMINISTIC_NODES
    )
    out["total_compute_s"] = total_time
    out["total_cost_usd"] = total_cost
    out["llm_share"] = (llm_time / total_time) if total_time else 0.0
    return out


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """mean ± std across runs, per node, for time and cost.

    ``runs`` is a list of per-node dicts (the ``aggregate_run`` output, or a
    plain ``{node: {"time_s", "cost_usd"}}`` mapping). Run-total keys
    (``total_*``, ``llm_share``) are summarized too when present.
    """
    keys: set[str] = set()
    for r in runs:
        keys.update(r.keys())

    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        if isinstance(runs[0].get(key), dict):
            t_vals = [r[key]["time_s"] for r in runs if key in r]
            c_vals = [r[key]["cost_usd"] for r in runs if key in r]
            summary[key] = {
                "time_s_mean": statistics.fmean(t_vals) if t_vals else 0.0,
                "time_s_std": statistics.pstdev(t_vals) if len(t_vals) > 1 else 0.0,
                "cost_usd_mean": statistics.fmean(c_vals) if c_vals else 0.0,
                "cost_usd_std": statistics.pstdev(c_vals) if len(c_vals) > 1 else 0.0,
                "n": float(len(t_vals)),
            }
        else:  # scalar run-total (total_compute_s, total_cost_usd, llm_share)
            vals = [float(r[key]) for r in runs if key in r]
            summary[key] = {
                "mean": statistics.fmean(vals) if vals else 0.0,
                "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            }
    return summary
