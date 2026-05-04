"""Async background task: runs the LangGraph pipeline and feeds events to RunStore."""
import asyncio
import time
from typing import Any

_STREAM_TIMEOUT = 300.0  # seconds before a hung LLM call is declared a failure

import pandas as pd
from langgraph.types import Command

from api.services import run_store
from api.services.pipeline_helpers import (
    build_initial_state,
    parse_stream_event,
    reset_tool_start_times,
)
from mlops_agents.graphs.mlops_graph import graph


def run_evidently(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
    """Run Evidently DataDriftPreset and return a DriftReport-shaped dict."""
    from datetime import datetime, timezone

    from evidently import Report
    from evidently.presets import DataDriftPreset

    report = Report([DataDriftPreset()])
    result = report.run(reference_df, current_df)
    raw = result.dump_dict()

    metrics = raw.get("metrics", [])
    dataset_metric = next((m for m in metrics if m.get("metric") == "DatasetDriftMetric"), {})
    column_metrics = [m for m in metrics if m.get("metric") == "ColumnDriftMetric"]

    drift_res = dataset_metric.get("result", {})
    columns = [
        {
            "column": m["result"].get("column_name", ""),
            "drift_detected": m["result"].get("drift_detected", False),
            "score": m["result"].get("drift_score", 0.0),
            "method": m["result"].get("stattest_name", ""),
        }
        for m in column_metrics
        if "result" in m
    ]

    drifted = sum(1 for c in columns if c["drift_detected"])
    drift_share = drifted / len(columns) if columns else 0.0

    return {
        "dataset_drift": drift_res.get("dataset_drift", False),
        "drift_share": drift_share,
        "columns": columns,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


async def pipeline_task(run_id: str, dataset_paths: list[str], schema_json: str = "") -> None:
    """Execute the LangGraph pipeline as an asyncio background task."""
    entry = run_store.get_entry(run_id)
    if entry is None:
        return

    from mlops_agents.config.settings import settings

    info_event: dict = {
        "type": "run_info",
        "agent": "system",
        "timestamp_ms": time.time() * 1000,
        "data": {
            "models": {
                "supervisor":     settings.openai_model_supervisor,
                "data_validator": settings.openai_model_data_validator,
                "trainer":        settings.openai_model_trainer,
                "evaluator":      settings.openai_model_evaluator,
                "deployer":       settings.openai_model_deployer,
            }
        },
    }
    entry.events.append(info_event)
    await entry.queue.put(info_event)

    reset_tool_start_times()
    initial_state = build_initial_state(dataset_paths, schema_json=schema_json)
    config = entry.graph_config

    # Accumulate per-agent reasoning tokens; emit one event per complete LLM turn.
    _reasoning_buf: dict[str, str] = {}

    async def _flush_reasoning() -> None:
        for agent_name, content in list(_reasoning_buf.items()):
            if content:
                ev: dict = {
                    "type": "agent_reasoning",
                    "agent": agent_name,
                    "timestamp_ms": time.time() * 1000,
                    "data": {"content": content},
                }
                entry.events.append(ev)
                await entry.queue.put(ev)
        _reasoning_buf.clear()

    async def _stream(source: Any) -> None:
        async for chunk in graph.astream(
            source, config, stream_mode=["updates", "messages"], subgraphs=True
        ):
            namespace, mode, data = chunk

            if mode == "updates":
                await _flush_reasoning()

                if "__interrupt__" in data:
                    interrupt_list = data["__interrupt__"]
                    interrupt_val = interrupt_list[0].value if interrupt_list else {}
                    entry.status = "awaiting_approval"
                    entry.interrupt_value = interrupt_val
                    hitl_agent = interrupt_val.get("type", "deployer")
                    hitl_event: dict = {
                        "type": "hitl_request",
                        "agent": hitl_agent,
                        "timestamp_ms": time.time() * 1000,
                        "data": interrupt_val,
                    }
                    entry.events.append(hitl_event)
                    await entry.queue.put(hitl_event)
                    return  # exit loop; wait for approval below

                if "supervisor" in data and isinstance(data["supervisor"], dict):
                    next_agent = data["supervisor"].get("next", "")
                    reasoning = data["supervisor"].get("reasoning", "")
                    if next_agent and next_agent != "FINISH":
                        event = {
                            "type": "routing",
                            "agent": "supervisor",
                            "timestamp_ms": time.time() * 1000,
                            "data": {"next": next_agent, "reasoning": reasoning},
                        }
                        entry.events.append(event)
                        await entry.queue.put(event)

            elif mode == "messages":
                pipeline_event = parse_stream_event(data)
                if pipeline_event:
                    if pipeline_event["type"] == "agent_reasoning":
                        agent = pipeline_event["agent"]
                        _reasoning_buf[agent] = _reasoning_buf.get(agent, "") + str(
                            pipeline_event["data"].get("content", "")
                        )
                    else:
                        await _flush_reasoning()
                        entry.events.append(dict(pipeline_event))
                        await entry.queue.put(dict(pipeline_event))

        await _flush_reasoning()

    async def _emit_error(msg: str) -> None:
        ev: dict = {
            "type": "run_complete",
            "agent": "supervisor",
            "timestamp_ms": time.time() * 1000,
            "data": {"error": msg},
        }
        entry.events.append(ev)
        await entry.queue.put(ev)
        entry.status = "failed"

    try:
        await asyncio.wait_for(_stream(initial_state), timeout=_STREAM_TIMEOUT)

        while entry.status == "awaiting_approval":
            entry.hitl_event.clear()
            await entry.hitl_event.wait()
            entry.status = "running"
            resume = {
                "approved": entry.hitl_decision == "approve",
                "comment": entry.hitl_comment,
            }
            await asyncio.wait_for(
                _stream(Command(resume=resume)), timeout=_STREAM_TIMEOUT
            )

        # Automatic drift detection is not performed here because the graph
        # state does not expose two distinct DataFrames (reference vs current).
        # Use POST /monitoring/drift for ad-hoc drift analysis.
        complete_event: dict = {
            "type": "run_complete",
            "agent": "supervisor",
            "timestamp_ms": time.time() * 1000,
            "data": {},
        }
        entry.events.append(complete_event)
        await entry.queue.put(complete_event)
        entry.status = "complete"

    except asyncio.TimeoutError:
        await _emit_error("Pipeline timed out — the LLM API may be unresponsive")

    except Exception as exc:
        await _emit_error(str(exc))
