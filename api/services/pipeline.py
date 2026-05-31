"""Async background task: runs the LangGraph pipeline and feeds events to RunStore."""
import asyncio
import time
from typing import Any

_STREAM_TIMEOUT = 300.0  # seconds before a hung LLM call is declared a failure

from langgraph.types import Command

from api.services import run_store
from api.services.pipeline_helpers import (
    build_initial_state,
    parse_stream_event,
    reset_tool_start_times,
)
from mlops_agents.graphs.mlops_graph import graph
from mlops_agents.agents.taxonomy import NODE_CATEGORIES
from mlops_agents.observability.pricing import estimate_cost


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
            # Similarity + tier — required by the frontend ExperienceCard
            "similarity_score": float(exp.get("similarity_score") or 0.0),
            "relevance_tier": exp.get("relevance_tier") or "low",
            "matched_buckets": exp.get("matched_buckets") or [],
            "mismatched_buckets": exp.get("mismatched_buckets") or [],
            "target_scale_note": exp.get("target_scale_note"),
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

    plan_summary_raw = rec.get("plan_summary", {}) or {}
    plan_summary = {
        # legacy string lists — keep for backward UI compat
        "candidate_models": plan_summary_raw.get("candidate_models", []),
        "models_not_recommended": plan_summary_raw.get("models_not_recommended", []),
        # v2 canonical keys — full rationale objects
        "candidate_rationales": plan_summary_raw.get("candidate_rationales", []),
        "rejected_model_rationales": plan_summary_raw.get("rejected_model_rationales", []),
    }

    decision_basis = rec.get("decision_basis") or {
        "primary_evidence": [],
        "secondary_evidence": [],
        "final_strategy": "",
    }

    return {
        "retrieved_experiences": retrieved_experiences,
        "matched_rules": matched_rules,
        "evidence_used": evidence_used,
        "planning_analysis": rec.get("planning_analysis", ""),
        "plan_summary": plan_summary,
        "warnings": rec.get("risks_or_warnings", []),
        # v2 fields
        "decision_basis": decision_basis,
        "evidence_conflicts": rec.get("evidence_conflicts", []) or [],
        "soft_conflicts": rec.get("soft_conflicts", []) or [],
        "cited_experience_ids": rec.get("cited_experience_ids", []) or [],
        "cited_rule_ids": rec.get("cited_rule_ids", []) or [],
        "planner_status": rec.get("planner_status", "ok"),
    }


async def pipeline_task(run_id: str, dataset_paths: list[str], schema_json: str = "") -> None:
    """Execute the LangGraph pipeline as an asyncio background task."""
    entry = run_store.get_entry(run_id)
    if entry is None:
        return

    from mlops_agents.config.settings import settings

    # Maps semantic node name → its configured model (used to fix fallback when response_metadata lacks model_name)
    node_model_map: dict[str, str] = {
        "data_validator": settings.openai_model_data_validator,
        "planner":        settings.openai_model_planner,
        "report_writer":  settings.openai_model_report_writer,
    }

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
            "node_categories": {
                "agents":        NODE_CATEGORIES["agents"],
                "llm_nodes":     NODE_CATEGORIES["llm_nodes"],
                "deterministic": NODE_CATEGORIES["deterministic"],
                "hitl":          NODE_CATEGORIES["hitl"],
            },
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

                if "deployer" in data and isinstance(data["deployer"], dict):
                    dp = data["deployer"]
                    if dp.get("deployment_status") == "deployed":
                        deploy_event: dict = {
                            "type": "deployment_complete",
                            "agent": "deployer",
                            "timestamp_ms": time.time() * 1000,
                            "data": {
                                "best_model_uri":     dp.get("best_model_uri", ""),
                                "deployment_status":  "deployed",
                            },
                        }
                        entry.events.append(deploy_event)
                        await entry.queue.put(deploy_event)

                if "executor" in data and isinstance(data["executor"], dict):
                    ex = data["executor"]
                    if ex.get("training_run_id") or ex.get("training_metrics") or ex.get("champion_candidate"):
                        training_event: dict = {
                            "type": "training_complete",
                            "agent": "executor",
                            "timestamp_ms": time.time() * 1000,
                            "data": {
                                "training_run_id":    ex.get("training_run_id", ""),
                                "training_metrics":   ex.get("training_metrics", {}),
                                "champion_candidate": ex.get("champion_candidate", {}),
                                "trained_model_path": ex.get("trained_model_path", ""),
                            },
                        }
                        entry.events.append(training_event)
                        await entry.queue.put(training_event)

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
                # Recover semantic node name from LangGraph namespace.
                # Inside a react agent, langgraph_node = "model"; namespace[0] = "data_validator:run_id".
                node_hint: str | None = None
                if namespace:
                    candidate = namespace[0].split(":")[0]
                    if candidate in node_model_map:
                        node_hint = candidate
                pipeline_event = parse_stream_event(data, node_hint=node_hint)
                if pipeline_event:
                    # Fix missing model name: when response_metadata lacked model_name,
                    # the event has model="" — fill in the correct model from settings.
                    if pipeline_event["type"] == "token_usage":
                        ev_data = pipeline_event["data"]
                        node_val: str = str(ev_data.get("node", ""))
                        model_val: str = str(ev_data.get("model", ""))
                        if not model_val and node_val in node_model_map:
                            correct_model = node_model_map[node_val]
                            ev_data["model"] = correct_model
                            ev_data["estimated_cost_usd"] = estimate_cost(
                                correct_model,
                                int(ev_data.get("input_tokens") or 0),
                                int(ev_data.get("output_tokens") or 0),
                                int(ev_data.get("cached_input_tokens") or 0),
                            )
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
