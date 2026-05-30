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


def _build_planner_ctx_event(rec: dict[str, Any]) -> dict[str, Any]:
    """Transform _planner_output_record to the shape the frontend PlannerContextData expects.

    Backend uses EvidenceReference field names (source/source_id/summary);
    frontend expects (evidence_type/experience_id|rule_id/relevance_note).
    Backend ExperienceSummary has dataset_summary; frontend expects dataset_name.
    """
    # evidence_used: {source, source_id, summary} → {evidence_type, experience_id|rule_id, relevance_note}
    evidence_used: list[dict[str, Any]] = []
    for ev in rec.get("evidence_used", []):
        source = ev.get("source", "")
        source_id = ev.get("source_id")
        item: dict[str, Any] = {
            "evidence_type": source if source in ("experience", "rule") else "rule",
            "relevance_note": ev.get("summary", ""),
        }
        if source == "experience":
            item["experience_id"] = source_id
        elif source == "rule":
            item["rule_id"] = source_id
        evidence_used.append(item)

    # retrieved_experiences: ExperienceSummary.model_dump() → frontend ExperienceSummary
    retrieved_experiences: list[dict[str, Any]] = [
        {
            "experience_id": exp.get("experience_id", ""),
            "dataset_name": exp.get("dataset_summary", exp.get("dataset_name", "")),
            "problem_type": exp.get("problem_type", ""),
            "best_model": exp.get("best_model", ""),
            # Guard against None so frontend .toFixed(4) never throws
            "validation_score": float(exp.get("validation_score") or 0.0),
            "metric_name": exp.get("metric_name"),
        }
        for exp in rec.get("retrieved_experiences", [])
    ]

    # matched_rules: coerce recommend (dict in Python) to a string so React can render it
    matched_rules: list[dict[str, Any]] = []
    for rule in rec.get("matched_rules", []):
        recommend_raw = rule.get("recommend")
        if isinstance(recommend_raw, dict):
            recommend_str: str | None = (
                ", ".join(f"{k}: {v}" for k, v in recommend_raw.items()) or None
            )
        elif isinstance(recommend_raw, str) and recommend_raw.strip():
            recommend_str = recommend_raw
        else:
            recommend_str = None
        matched_rules.append({**rule, "recommend": recommend_str})

    return {
        "retrieved_experiences": retrieved_experiences,
        "matched_rules": matched_rules,
        "evidence_used": evidence_used,
        "planning_analysis": rec.get("planning_analysis", ""),
        "plan_summary": rec.get("plan_summary", {}),
        "warnings": rec.get("risks_or_warnings", []),
    }


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

    # Read problem_type from the schema JSON the caller posted
    import json as _json
    pt = ""
    try:
        pt = _json.loads(schema_json or "{}").get("problem_type", "")
    except Exception:
        pt = ""

    info_event: dict = {
        "type": "run_info",
        "agent": "system",
        "timestamp_ms": time.time() * 1000,
        "data": {
            "models": {
                "data_validator": settings.openai_model_data_validator,
                "planner":        settings.openai_model_planner,
                "report_writer":  settings.openai_model_report_writer,
            },
            "problem_type": pt,
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

                    if hitl_agent == "data_validation":
                        preview = (
                            interrupt_val.get("dataset_preview")
                            or interrupt_val.get("preview")
                            or {}
                        )
                        path = preview.get("path")
                        if isinstance(path, str) and path:
                            entry.processed_dataset_path = path

                    hitl_event: dict = {
                        "type": "hitl_request",
                        "agent": hitl_agent,
                        "timestamp_ms": time.time() * 1000,
                        "data": interrupt_val,
                    }
                    entry.events.append(hitl_event)
                    await entry.queue.put(hitl_event)
                    return  # exit loop; wait for approval below

                if "planner" in data and isinstance(data["planner"], dict):
                    rec = data["planner"].get("_planner_output_record") or {}
                    if rec:
                        planner_ctx_event: dict = {
                            "type": "planner_context",
                            "agent": "planner",
                            "timestamp_ms": time.time() * 1000,
                            "data": _build_planner_ctx_event(rec),
                        }
                        entry.events.append(planner_ctx_event)
                        await entry.queue.put(planner_ctx_event)

                if "report_writer" in data and isinstance(data["report_writer"], dict):
                    rw = data["report_writer"]
                    audit = rw.get("evaluation_report_audit")
                    if isinstance(audit, dict) and audit:
                        from mlops_agents.evaluation.champion import resolve_champion_model_name
                        # evaluation_passed lives INSIDE the audit dict
                        # (report_writer.py copies it through from prior state)
                        evaluation_passed = bool(audit.get("evaluation_passed", True))
                        champion = resolve_champion_model_name({**rw})
                        audit_event: dict = {
                            "type": "audit_report",
                            "agent": "report_writer",
                            "timestamp_ms": time.time() * 1000,
                            "data": {
                                "audit":              audit,
                                "champion_model":     audit.get("champion_model") or champion,
                                "evaluation_passed":  evaluation_passed,
                                "candidate_metrics":  rw.get("candidate_metrics", {}),
                                "champion_metrics":   rw.get("champion_metrics", {}),
                                "thresholds_applied": rw.get("thresholds_applied", {}),
                            },
                        }
                        entry.events.append(audit_event)
                        await entry.queue.put(audit_event)

                worker_nodes = {
                    "data_validator", "dataset_approval", "planner",
                    "executor", "evaluation", "report_writer",
                    "deployment_approval", "deployer",
                }
                for node_name in data:
                    if node_name in worker_nodes:
                        event = {
                            "type": "routing",
                            "agent": "controller",  # UI label, preserved for FE
                            "timestamp_ms": time.time() * 1000,
                            "data": {"next": node_name, "reasoning": ""},
                        }
                        entry.events.append(event)
                        await entry.queue.put(event)
                        break

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
            "agent": "controller",
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
            "agent": "controller",
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
