# Observability & Logging Design

**Date:** 2026-04-22
**Status:** Approved

## Problem

The current logging setup emits only coarse-grained node completion events (`[data_validator] completed`). Tool arguments, LLM reasoning text, tool results, and timing are invisible. `mlops_graph.py` still uses raw `print()` calls in `main()` and the HITL handler. There is no way to inspect what happened inside an agent run after the fact.

## Goal

Surface a structured, filterable trace of every pipeline run — tool calls with their arguments and results, agent reasoning, supervisor routing decisions, and durations — in a dedicated Streamlit Logs page. Verbosity is configurable so the view can be dialled from a high-level summary to a full trace without re-running the pipeline.

---

## Architecture

### Event capture

`graph.stream()` in `dashboard/pages/01_pipeline.py` switches to:

```python
graph.stream(initial_state, config, stream_mode=["updates", "messages"], subgraphs=True)
```

LangGraph yields two interleaved chunk types:

- `("updates", {...})` — node completion events. **Existing Pipeline page live-log behavior is unchanged** — these are processed exactly as before.
- `("messages", (chunk, metadata))` — every message produced inside each agent subgraph: AIMessage reasoning text, tool call decisions (name + arguments), ToolMessage results.

The Pipeline page additionally passes `"messages"` chunks to `parse_stream_event()` and appends the returned `PipelineEvent` dicts to `st.session_state["run_events"]`. This list is cleared at the start of each new run.

The Logs page (`05_logs.py`) reads `st.session_state["run_events"]` — no graph re-execution needed.

---

## Event Model

Each event is a `PipelineEvent` TypedDict:

```python
class PipelineEvent(TypedDict):
    type: str          # "routing" | "tool_call" | "tool_result" | "agent_reasoning"
    agent: str         # "supervisor" | "data_validator" | "trainer" | "evaluator" | "deployer"
    timestamp_ms: float
    data: dict         # varies by type (see below)
```

### `data` per event type

| `type` | `data` keys |
|--------|-------------|
| `routing` | `next: str`, `reasoning: str` |
| `tool_call` | `tool_name: str`, `arguments: dict` |
| `tool_result` | `tool_name: str`, `result: dict \| str`, `duration_ms: float` |
| `agent_reasoning` | `content: str` |

**Duration tracking:** `parse_stream_event` keeps an internal `dict[str, float]` mapping `tool_name → call_start_ms`. When a `tool_call` event is emitted the start time is recorded; when the matching `tool_result` arrives `duration_ms` is computed and the entry cleared.

**All events are always stored** regardless of verbosity level. Verbosity only controls rendering.

---

## Verbosity Config

New field in `src/mlops_agents/config/settings.py`:

```python
log_verbosity: int = 2  # overridable via LOG_VERBOSITY env var
```

| Level | Name | Event types rendered |
|-------|------|----------------------|
| 1 | Summary | `routing` |
| 2 | Standard | `routing`, `tool_call`, `tool_result` |
| 3 | Full trace | `routing`, `tool_call`, `tool_result`, `agent_reasoning` |

The Logs page exposes a verbosity dropdown (default: `settings.log_verbosity`) that overrides the setting for the current browser session, so you can re-filter an already-completed run without re-running the pipeline.

---

## Components

### `parse_stream_event(chunk: tuple) -> PipelineEvent | None`

Added to `dashboard/pipeline_helpers.py`. Receives one raw chunk from `graph.stream()` when `stream_mode` includes `"messages"`. Returns a `PipelineEvent` dict or `None` if the chunk is not actionable (e.g. partial token stream with no tool info yet).

Logic:
1. Extract `(message_chunk, metadata)` from the chunk.
2. Read `agent = metadata.get("langgraph_node", "unknown")`.
3. Classify:
   - `AIMessageChunk` with non-empty `content` and no `tool_calls` → `agent_reasoning`
   - `AIMessageChunk` with `tool_calls` → `tool_call` (record start time)
   - `ToolMessage` → `tool_result` (compute duration from recorded start time)
4. Return the structured event, or `None`.

### `dashboard/pages/05_logs.py` — new page

Controls (sidebar or top of page):
- **Verbosity** — selectbox: Summary / Standard / Full trace (default from `settings.log_verbosity`)
- **Agent filter** — multiselect: all agents checked by default
- **Event type filter** — checkboxes derived from verbosity selection

Timeline rendering:
- One `st.expander` per event, collapsed by default
- Title: `HH:MM:SS.mmm  [agent]  type — tool_name or →next_agent`
- Expanded: `st.json(event["data"])`
- When `st.session_state["run_events"]` is empty: `st.info("No events — run the pipeline first.")`

### `dashboard/pages/01_pipeline.py` — changes only in the run loop

```python
# Before streaming, clear previous run events
st.session_state["run_events"] = []

for chunk in graph.stream(build_initial_state(dataset_paths), config=config,
                          stream_mode=["updates", "messages"], subgraphs=True):
    mode, data = chunk
    if mode == "updates":
        # existing node-completion handling — unchanged
        ...
    elif mode == "messages":
        event = parse_stream_event(data)
        if event:
            st.session_state["run_events"].append(event)
```

Same change applied to the resume-after-HITL streaming loop in `_resume_pipeline`.

---

## Files Changed

| File | Change |
|------|--------|
| `src/mlops_agents/config/settings.py` | Add `log_verbosity: int = 2` |
| `dashboard/pipeline_helpers.py` | Add `PipelineEvent` TypedDict + `parse_stream_event()` |
| `dashboard/pages/01_pipeline.py` | Switch to multi-mode streaming, clear + populate `run_events` |
| `dashboard/pages/05_logs.py` | New — Logs page with verbosity + agent filtering |

---

## What Is Not Changed

- Graph topology, nodes, supervisor, agents, tools
- `AgentState` schema
- `mlops_graph.py` CLI (`main()`) and HITL handler — `print()` calls remain there (CLI context, not dashboard)
- Existing Pipeline page live-log behavior

---

## Testing

- `parse_stream_event` is a pure function — unit-testable with synthetic `(message_chunk, metadata)` tuples, no LangGraph needed.
- Test cases: AIMessage with content only → `agent_reasoning`; AIMessage with tool_calls → `tool_call`; ToolMessage → `tool_result` with duration; unknown chunk → `None`.
- Verbosity filter logic is also pure (filter a list of events) — tested separately.
