"""Headless agentic-pipeline cost/timing measurement harness.

Runs the FULL LangGraph pipeline (data_validator → planner → executor →
evaluation → report_writer → deployer) headless, with both HITL gates
auto-approved, N times per dataset. Each run's EventLog is dumped to JSON and
aggregated (per-node compute time + cost) by ``agentic_cost_aggregate``.

This is the reproducible replacement for the hand-built Chapter 6 cost table.
It mirrors the app's telemetry (same ``parse_stream_event`` + ``estimate_cost``)
so numbers match an app run in expectation, but isolates the human HITL pause
(auto-approve is immediate) and is meant to be run on an OTHERWISE-IDLE machine
to avoid the CPU-contention / network-stall artefacts that inflated the
previous measurement.

Usage:
    uv run python scripts/measure_agentic_cost.py --smoke              # bakery ×1
    uv run python scripts/measure_agentic_cost.py --datasets grid ×1   # one join ×1
    uv run python scripts/measure_agentic_cost.py --runs 5             # full sweep 6×5
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", category=FutureWarning)

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from langgraph.types import Command  # noqa: E402

from scripts.agentic_cost_aggregate import (  # noqa: E402
    WORKER_NODES,
    aggregate_run,
    summarize_runs,
)

_S = "data/samples"

# name → (raw input files, posted schema_json file). Single-file forecasting
# posts the per-series schema; join datasets post the joined-target schema and
# upload all raw source files (incl. decoys) so join discovery runs for real.
DATASETS: dict[str, dict[str, Any]] = {
    "revenue": {
        "label": "Revenue (120, monthly)",
        "paths": [f"{_S}/size_test/small_monthly_revenue.csv"],
        "schema": f"{_S}/size_test/small_monthly_revenue_schema.json",
    },
    "bakery": {
        "label": "Bakery (1 000, daily)",
        "paths": [f"{_S}/size_test/medium_daily_bakery.csv"],
        "schema": f"{_S}/size_test/medium_daily_bakery_schema.json",
    },
    "large": {
        "label": "Large (10 000, hourly)",
        "paths": [f"{_S}/size_test/large_hourly_factory.csv"],
        "schema": f"{_S}/size_test/large_hourly_factory_schema.json",
    },
    "grid": {
        "label": "Grid (4-file join)",
        "paths": [
            f"{_S}/forecasting2/grid_demand.csv",
            f"{_S}/forecasting2/calendar_registry.csv",
            f"{_S}/forecasting2/generation_mix.csv",
            f"{_S}/forecasting2/meteorological.csv",
        ],
        "schema": f"{_S}/forecasting2/grid_demand_schema.json",
    },
    "airpassengers": {
        "label": "AirPassengers (3-file join)",
        "paths": [
            f"{_S}/join_discovery/air_passengers_base.csv",
            f"{_S}/join_discovery/aviation_context.csv",
            f"{_S}/join_discovery/product_catalog.csv",
        ],
        "schema": f"{_S}/join_discovery/air_passengers_joined_schema.json",
    },
    "retail": {
        "label": "Retail (4-file join)",
        "paths": [
            f"{_S}/retail_star_schema/sales_transactions.csv",
            f"{_S}/retail_star_schema/dim_customers.csv",
            f"{_S}/retail_star_schema/dim_products.csv",
            f"{_S}/retail_star_schema/dim_stores.csv",
            f"{_S}/retail_star_schema/dim_brands.csv",
        ],
        "schema": f"{_S}/retail_star_schema/retail_sales_schema.json",
    },
    "demand_ts": {
        "label": "Demand (3-file join, parcial)",
        "paths": [
            f"{_S}/join_partial_coverage_ts_single/weekly_demand.csv",
            f"{_S}/join_partial_coverage_ts_single/weather_weekly.csv",
            f"{_S}/join_partial_coverage_ts_single/calendar_weekly.csv",
        ],
        "schema": f"{_S}/join_partial_coverage_ts_single/weekly_demand_schema.json",
    },
}

_STREAM_TIMEOUT = 1000.0


def _now_ms() -> float:
    return time.time() * 1000


async def run_pipeline_once(paths: list[str], schema_json: str) -> list[dict[str, Any]]:
    """Drive one full pipeline run headless; return the collected EventLog.

    Auto-approves both HITL gates. Mirrors api/services/pipeline.py event
    shaping for the events the aggregator needs (run_info, routing, token_usage,
    run_complete) plus archival events (hitl_request/resolved, training_complete).
    """
    from mlops_agents.graphs.taxonomy import NODE_CATEGORIES
    from mlops_agents.config.settings import settings
    from mlops_agents.graphs.mlops_graph import graph
    from mlops_agents.observability.pricing import estimate_cost
    from mlops_agents.prompts.loader import get_agent_config
    from api.services.pipeline_helpers import (
        build_initial_state,
        parse_stream_event,
        reset_tool_start_times,
    )

    _agent_yaml = {"data_validator": "data_agent", "planner": "planner", "report_writer": "report_writer"}
    node_model_map = {
        node: get_agent_config(yaml).get("model", settings.openai_model)
        for node, yaml in _agent_yaml.items()
    }

    pt = ""
    try:
        pt = json.loads(schema_json or "{}").get("problem_type", "")
    except Exception:
        pt = ""

    events: list[dict[str, Any]] = []
    events.append({
        "type": "run_info", "agent": "system", "timestamp_ms": _now_ms(),
        "data": {
            "models": node_model_map, "problem_type": pt,
            "node_categories": {k: NODE_CATEGORIES[k] for k in
                                ("agents", "llm_nodes", "deterministic", "hitl")},
        },
    })

    reset_tool_start_times()
    initial_state = build_initial_state(paths, schema_json=schema_json)
    config = {"configurable": {"thread_id": uuid.uuid4().hex}, "recursion_limit": 50}

    worker_nodes = {
        "data_validator", "dataset_approval", "planner", "executor",
        "evaluation", "report_writer", "deployment_approval", "deployer",
    }
    awaiting = {"flag": False}

    async def _stream(source: Any) -> None:
        async for chunk in graph.astream(
            source, config, stream_mode=["updates", "messages", "custom"], subgraphs=True
        ):
            _ns, mode, data = chunk
            if mode == "custom":
                # Planner emits validation-error / retry signals from inside its node
                # via get_stream_writer(); capture them to diagnose failures.
                if isinstance(data, dict) and data.get("kind") in (
                    "planner_validation_error", "planner_retry"
                ):
                    events.append({
                        "type": data["kind"], "agent": "planner",
                        "timestamp_ms": _now_ms(),
                        "data": {k: v for k, v in data.items() if k != "kind"},
                    })
                continue
            if mode == "updates":
                if "__interrupt__" in data:
                    iv = data["__interrupt__"][0].value if data["__interrupt__"] else {}
                    events.append({
                        "type": "hitl_request", "agent": iv.get("type", "deployer"),
                        "timestamp_ms": _now_ms(), "data": {"type": iv.get("type", "")},
                    })
                    awaiting["flag"] = True
                    return
                if "planner" in data and isinstance(data["planner"], dict):
                    rec = data["planner"].get("_planner_output_record") or {}
                    if rec:
                        cands = (rec.get("plan_summary") or {}).get("candidate_models") or []
                        events.append({
                            "type": "planner_context", "agent": "planner",
                            "timestamp_ms": _now_ms(),
                            "data": {"candidate_models": cands, "n_candidates": len(cands)},
                        })
                if "executor" in data and isinstance(data["executor"], dict):
                    ex = data["executor"]
                    if ex.get("training_run_id") or ex.get("champion_candidate"):
                        events.append({
                            "type": "training_complete", "agent": "executor",
                            "timestamp_ms": _now_ms(),
                            "data": {"champion_candidate": ex.get("champion_candidate", {})},
                        })
                for node_name in data:
                    if node_name in worker_nodes:
                        events.append({
                            "type": "routing", "agent": "controller",
                            "timestamp_ms": _now_ms(),
                            "data": {"next": node_name, "reasoning": ""},
                        })
                        break
            elif mode == "messages":
                node_hint = None
                if _ns:
                    cand = _ns[0].split(":")[0]
                    if cand in node_model_map:
                        node_hint = cand
                ev = parse_stream_event(data, node_hint=node_hint)
                if ev and ev["type"] == "token_usage":
                    d = ev["data"]
                    if not d.get("model") and d.get("node") in node_model_map:
                        m = node_model_map[d["node"]]
                        d["model"] = m
                        d["estimated_cost_usd"] = estimate_cost(
                            m, int(d.get("input_tokens") or 0),
                            int(d.get("output_tokens") or 0),
                            int(d.get("cached_input_tokens") or 0),
                        )
                    events.append(dict(ev))

    await asyncio.wait_for(_stream(initial_state), timeout=_STREAM_TIMEOUT)
    while awaiting["flag"]:
        awaiting["flag"] = False
        events.append({
            "type": "hitl_resolved", "agent": "controller", "timestamp_ms": _now_ms(),
            "data": {"decision": "approve", "comment": ""},
        })
        await asyncio.wait_for(
            _stream(Command(resume={"approved": True, "comment": ""})),
            timeout=_STREAM_TIMEOUT,
        )

    events.append({"type": "run_complete", "agent": "controller",
                   "timestamp_ms": _now_ms(), "data": {}})
    return events


async def _sweep(names: list[str], n_runs: int, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_summaries: dict[str, Any] = {}

    for name in names:
        spec = DATASETS[name]
        schema_json = Path(spec["schema"]).read_text(encoding="utf-8")
        per_run: list[dict[str, Any]] = []
        print(f"\n=== {name} ({spec['label']}) — {n_runs} run(s) ===", flush=True)

        for i in range(1, n_runs + 1):
            t0 = time.time()
            try:
                events = await run_pipeline_once(spec["paths"], schema_json)
            except Exception as exc:  # noqa: BLE001 — one bad run must not kill the sweep
                print(f"  run {i}: FAILED ({type(exc).__name__}: {exc})", flush=True)
                continue
            (out_dir / f"{name}_run{i}.json").write_text(
                json.dumps(events, indent=2), encoding="utf-8"
            )
            agg = aggregate_run(events)
            per_run.append(agg)
            wall = time.time() - t0
            print(f"  run {i}: compute={agg['total_compute_s']:.1f}s "
                  f"cost=${agg['total_cost_usd']:.3f} llm={agg['llm_share']*100:.0f}% "
                  f"(wall {wall:.0f}s)", flush=True)

        if per_run:
            summary = summarize_runs(per_run)
            all_summaries[name] = {"label": spec["label"], "n": len(per_run), "summary": summary}
            (out_dir / f"{name}_summary.json").write_text(
                json.dumps(all_summaries[name], indent=2), encoding="utf-8"
            )

    (out_dir / "all_summaries.json").write_text(
        json.dumps(all_summaries, indent=2), encoding="utf-8"
    )
    _print_table(all_summaries)
    return all_summaries


def _print_table(summaries: dict[str, Any]) -> None:
    print("\n================ PER-NODE TIME (s) · COST ($) — mean ================")
    print(f"{'node':<16}" + "".join(f"{summaries[n]['label'][:14]:>16}" for n in summaries))
    for node in WORKER_NODES:
        row = f"{node:<16}"
        for n in summaries:
            s = summaries[n]["summary"].get(node)
            cell = "—" if not s else f"{s['time_s_mean']:.1f}·${s['cost_usd_mean']:.3f}"
            row += f"{cell:>16}"
        print(row)
    for total in ("total_compute_s", "total_cost_usd", "llm_share"):
        row = f"{total:<16}"
        for n in summaries:
            s = summaries[n]["summary"].get(total, {})
            row += f"{s.get('mean', 0):>16.3f}"
        print(row)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="", help="comma list; default = all 6")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--smoke", action="store_true", help="bakery ×1 (cheap end-to-end check)")
    p.add_argument("--out", type=Path, default=Path("data/benchmarks/cost_runs"))
    args = p.parse_args()

    if args.smoke:
        names, n_runs = ["bakery"], 1
    else:
        names = [s.strip() for s in args.datasets.split(",") if s.strip()] or list(DATASETS)
        n_runs = args.runs

    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        sys.exit(f"unknown datasets: {unknown}; valid = {list(DATASETS)}")

    asyncio.run(_sweep(names, n_runs, args.out))


if __name__ == "__main__":
    main()
