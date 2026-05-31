"""Pipeline helper functions: build initial state, parse stream events."""

import time
from typing import TypedDict

from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage

from mlops_agents.observability.pricing import estimate_cost

_tool_start_times: dict[str, float] = {}


def reset_tool_start_times() -> None:
    _tool_start_times.clear()


class PipelineEvent(TypedDict):
    type: str
    agent: str
    timestamp_ms: float
    data: dict


def build_initial_state(dataset_paths: list[str], schema_json: str = "") -> dict:
    paths_display = ", ".join(dataset_paths)
    return {
        "messages": [HumanMessage(content=f"Run the full MLOps pipeline on these raw files: {paths_display}")],
        "next": "",
        "dataset_paths": dataset_paths,
        "processed_dataset_path": "",
        "schema_json": schema_json,
        "dataset_summary": {},
        "problem_type": "",
        "task_metadata": {},
        "validation_passed": False,
        "validation_report": {},
        "trained_model_path": "",
        "training_run_id": "",
        "training_metrics": {},
        "evaluation_passed": None,
        "evaluation_report": {},
        "best_model_uri": "",
        "deployment_decision": "pending",
        "deployment_status": "",
        "error_message": "",
        "agent_attempt_counts": {},
        # Refactor additions
        "dataset_approved": None,
        "dataset_rejection_comment": "",
        "deployment_approved": None,
        "candidate_metrics": {},
        "champion_metrics": {},
        "thresholds_applied": {},
        "evaluation_report_audit": None,
        "evaluation_report_audit_status": "",
    }


def parse_stream_event(chunk: tuple) -> PipelineEvent | None:
    try:
        message_chunk, metadata = chunk
    except (TypeError, ValueError):
        return None

    agent: str = metadata.get("langgraph_node", "unknown") if isinstance(metadata, dict) else "unknown"
    now_ms: float = time.time() * 1000

    if isinstance(message_chunk, AIMessageChunk):
        tool_calls = message_chunk.tool_calls
        if tool_calls:
            tool_name: str = tool_calls[0]["name"]
            _tool_start_times[tool_name] = now_ms
            return PipelineEvent(
                type="tool_call",
                agent=agent,
                timestamp_ms=now_ms,
                data={"tool_name": tool_name, "arguments": tool_calls[0].get("args", {})},
            )
        elif message_chunk.content:
            return PipelineEvent(
                type="agent_reasoning",
                agent=agent,
                timestamp_ms=now_ms,
                data={"content": message_chunk.content},
            )
        elif message_chunk.usage_metadata:
            usage = message_chunk.usage_metadata
            model_name: str = (
                (message_chunk.response_metadata or {}).get("model_name", "")
                or agent
            )
            input_t: int = usage.get("input_tokens", 0)
            output_t: int = usage.get("output_tokens", 0)
            cached_t: int = (usage.get("input_token_details") or {}).get("cache_read", 0)
            return PipelineEvent(
                type="token_usage",
                agent=agent,
                timestamp_ms=now_ms,
                data={
                    "node": agent,
                    "model": model_name,
                    "input_tokens": input_t,
                    "output_tokens": output_t,
                    "total_tokens": usage.get("total_tokens", input_t + output_t),
                    "cached_input_tokens": cached_t if cached_t else None,
                    "estimated_cost_usd": estimate_cost(model_name, input_t, output_t, cached_t),
                    "source": "langchain_stream_usage_metadata",
                },
            )
        return None

    if isinstance(message_chunk, ToolMessage):
        tool_name = message_chunk.name or ""
        start_ms = _tool_start_times.pop(tool_name, now_ms)
        duration_ms: float = now_ms - start_ms
        return PipelineEvent(
            type="tool_result",
            agent=agent,
            timestamp_ms=now_ms,
            data={"tool_name": tool_name, "result": message_chunk.content, "duration_ms": duration_ms},
        )

    return None
