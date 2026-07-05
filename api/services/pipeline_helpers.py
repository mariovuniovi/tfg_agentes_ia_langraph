"""Pipeline helper functions: build initial state, parse stream events."""

import time
from typing import TypedDict

from langchain_core.messages import AIMessageChunk, HumanMessage, ToolMessage

from mlops_agents.observability.pricing import estimate_cost, normalize

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
        "data_join_plan": None,
        "data_join_base_nrows": None,
        "data_join_evaluations": [],
    }


_LANGRAPH_INTERNAL_NODES = frozenset({"model", "tools", "__start__", "unknown"})


def _extract_reasoning_text(blocks: list) -> str:
    """Extract reasoning summary text from responses/v1 content blocks."""
    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "reasoning":
            for summary in block.get("summary", []):
                if isinstance(summary, dict) and summary.get("text"):
                    parts.append(summary["text"])
    return " ".join(parts)


def _extract_output_text(blocks: list) -> str:
    """Extract plain output text from responses/v1 content blocks."""
    parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type", "")
        if btype in ("output_text", "text"):
            text = block.get("text", "")
            if text:
                parts.append(text)
    return "".join(parts)


def parse_stream_event(
    chunk: tuple,
    node_hint: str | None = None,
) -> PipelineEvent | None:
    try:
        message_chunk, metadata = chunk
    except (TypeError, ValueError):
        return None

    raw_node: str = metadata.get("langgraph_node", "unknown") if isinstance(metadata, dict) else "unknown"
    # LangGraph uses "model"/"tools" as internal node names inside react agents.
    # Use the parent namespace hint to recover the semantic node name.
    agent: str = node_hint if (raw_node in _LANGRAPH_INTERNAL_NODES and node_hint) else raw_node
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
            content = message_chunk.content
            # responses/v1 format: content is a list of typed blocks
            if isinstance(content, list):
                reasoning_text = _extract_reasoning_text(content)
                output_text = _extract_output_text(content)
                if reasoning_text:
                    return PipelineEvent(
                        type="agent_thinking",
                        agent=agent,
                        timestamp_ms=now_ms,
                        data={"content": reasoning_text},
                    )
                if output_text:
                    return PipelineEvent(
                        type="agent_reasoning",
                        agent=agent,
                        timestamp_ms=now_ms,
                        data={"content": output_text},
                    )
                return None
            return PipelineEvent(
                type="agent_reasoning",
                agent=agent,
                timestamp_ms=now_ms,
                data={"content": content},
            )
        elif message_chunk.usage_metadata:
            usage = message_chunk.usage_metadata
            # Normalize: strip provider prefix and date suffix (e.g. "openai/gpt-5.4-mini-2026-03-17" → "gpt-5.4-mini")
            raw_model: str = (message_chunk.response_metadata or {}).get("model_name", "") or ""
            model_name: str = normalize(raw_model) if raw_model else ""
            input_t: int = usage.get("input_tokens", 0)
            output_t: int = usage.get("output_tokens", 0)
            cached_t: int = (usage.get("input_token_details") or {}).get("cache_read", 0)
            reasoning_t: int = (usage.get("output_token_details") or {}).get("reasoning", 0)
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
                    "reasoning_tokens": reasoning_t if reasoning_t else None,
                    "estimated_cost_usd": estimate_cost(model_name, input_t, output_t, cached_t) if model_name else None,
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
