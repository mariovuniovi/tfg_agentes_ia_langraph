"""Logs page — interactive timeline of pipeline events."""

import sys
from datetime import datetime
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parents[2]))

from mlops_agents.config.settings import settings

st.set_page_config(page_title="Logs", layout="wide")
st.title("📋 Pipeline Logs")

run_events = st.session_state.get("run_events", [])

if not run_events:
    st.info("No events — run the pipeline first.")
    st.stop()

col1, col2 = st.columns([2, 4])

with col1:
    verbosity_idx = settings.log_verbosity - 1
    verbosity_idx = max(0, min(2, verbosity_idx))
    verbosity_options = ["Summary", "Standard", "Full trace"]
    selected_verbosity = st.selectbox("Verbosity", options=verbosity_options, index=verbosity_idx)
    verbosity_map = {"Summary": 1, "Standard": 2, "Full trace": 3}
    selected_level = verbosity_map[selected_verbosity]

with col2:
    unique_agents = sorted(set(e.get("agent", "unknown") for e in run_events if e))
    default_agents = unique_agents if unique_agents else ["supervisor", "data_validator", "trainer", "evaluator", "deployer"]
    selected_agents = st.multiselect("Agents", options=default_agents, default=default_agents)

type_filter_map = {
    1: {"routing"},
    2: {"routing", "tool_call", "tool_result"},
    3: {"routing", "tool_call", "tool_result", "agent_reasoning"},
}
allowed_types = type_filter_map[selected_level]

filtered_events = [
    e for e in run_events
    if e.get("type") in allowed_types and e.get("agent") in selected_agents
]

if not filtered_events:
    st.info("No events match the current filters.")
    st.stop()

for event in filtered_events:
    ts_ms = event.get("timestamp_ms", 0)
    dt = datetime.fromtimestamp(ts_ms / 1000)
    time_str = dt.strftime("%H:%M:%S.") + f"{int(ts_ms) % 1000:03d}"

    agent = event.get("agent", "unknown")
    event_type = event.get("type", "unknown")
    data = event.get("data", {})

    if event_type == "routing":
        detail = f"→ {data.get('next', '?')}"
    elif event_type in ("tool_call", "tool_result"):
        detail = data.get("tool_name", "?")
    elif event_type == "agent_reasoning":
        content = data.get("content", "")
        detail = content[:60].replace("\n", " ")
    else:
        detail = ""

    title = f"{time_str}  [{agent}]  {event_type}"
    if detail:
        title += f" — {detail}"

    with st.expander(title, expanded=False):
        st.json(data)
