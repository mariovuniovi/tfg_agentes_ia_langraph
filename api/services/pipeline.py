"""Async background task: runs the LangGraph pipeline and feeds events to RunStore."""
import time
from typing import Any

import pandas as pd
from langgraph.types import Command

from api.services import run_store
from dashboard.pipeline_helpers import (
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


async def pipeline_task(run_id: str, dataset_paths: list[str]) -> None:
    """Execute the LangGraph pipeline as an asyncio background task."""
    entry = run_store.get_entry(run_id)
    if entry is None:
        return

    reset_tool_start_times()
    initial_state = build_initial_state(dataset_paths)
    config = entry.graph_config

    async def _stream(source: Any) -> None:
        async for chunk in graph.astream(
            source, config, stream_mode=["updates", "messages"], subgraphs=True
        ):
            namespace, mode, data = chunk

            if mode == "updates":
                if "__interrupt__" in data:
                    interrupt_list = data["__interrupt__"]
                    interrupt_val = interrupt_list[0].get("value", {}) if interrupt_list else {}
                    entry.status = "awaiting_approval"
                    entry.interrupt_value = interrupt_val
                    hitl_event: dict = {
                        "type": "hitl_request",
                        "agent": "deployer",
                        "timestamp_ms": time.time() * 1000,
                        "data": interrupt_val,
                    }
                    entry.events.append(hitl_event)
                    await entry.queue.put(hitl_event)
                    return  # exit loop; wait for approval below

                if "supervisor" in data:
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
                    entry.events.append(dict(pipeline_event))
                    await entry.queue.put(dict(pipeline_event))

    try:
        await _stream(initial_state)

        if entry.status == "awaiting_approval":
            await entry.hitl_event.wait()
            entry.status = "running"
            await _stream(Command(resume=entry.hitl_decision))

        final_state = (await graph.aget_state(config)).values
        dataset_path = final_state.get("dataset_path", "")
        try:
            if dataset_path:
                current_df = pd.read_csv(dataset_path)
                reference_df = current_df
                drift_report = run_evidently(reference_df, current_df)
            else:
                drift_report = {}
        except Exception:
            drift_report = {}

        entry.last_drift_report = drift_report
        run_store.set_latest_drift_report(drift_report)

        complete_event = {
            "type": "run_complete",
            "agent": "supervisor",
            "timestamp_ms": time.time() * 1000,
            "data": drift_report,
        }
        entry.events.append(complete_event)
        await entry.queue.put(complete_event)
        entry.status = "complete"

    except Exception as exc:
        error_event = {
            "type": "run_complete",
            "agent": "supervisor",
            "timestamp_ms": time.time() * 1000,
            "data": {"error": str(exc)},
        }
        entry.events.append(error_event)
        await entry.queue.put(error_event)
        entry.status = "failed"
