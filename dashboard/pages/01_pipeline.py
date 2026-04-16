"""Pipeline page — run and monitor the MLOps agent pipeline.

Three-phase state machine stored in st.session_state:
  idle             → shows dataset selector + Run button
  awaiting_approval → shows frozen log + HITL approval panel
  complete         → shows full log + outcome banner + Run Again button
"""

import time
from pathlib import Path

import streamlit as st
from langgraph.types import Command

from dashboard.pipeline_helpers import build_initial_state, event_to_log_line

st.set_page_config(page_title="Pipeline", layout="wide")
st.title("🤖 MLOps Pipeline")

# ── Session state initialisation ──────────────────────────────────────────────
_DEFAULTS: dict = {
    "phase": "idle",            # idle | awaiting_approval | complete
    "log_lines": [],
    "pipeline_config": None,
    "interrupt_value": {},
    "deployment_decision": "pending",
    "final_message": "",
    "reject_mode": False,
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Helpers ───────────────────────────────────────────────────────────────────

def _log(line: str) -> None:
    st.session_state["log_lines"].append(line)


def _render_log(placeholder) -> None:
    placeholder.markdown("\n\n".join(st.session_state["log_lines"]))


def _resume_pipeline(resume: dict) -> None:
    """Resume the paused graph with the operator decision and stream remaining events."""
    from mlops_agents.graphs.mlops_graph import graph

    config = st.session_state["pipeline_config"]
    st.subheader("Pipeline Log")
    log_placeholder = st.empty()
    _render_log(log_placeholder)

    for event in graph.stream(Command(resume=resume), config=config):
        line = event_to_log_line(event)
        if line:
            _log(line)
            _render_log(log_placeholder)

    final = graph.get_state(config).values
    st.session_state["deployment_decision"] = final.get("deployment_decision", "pending")
    msgs = final.get("messages", [])
    if msgs:
        last = msgs[-1]
        st.session_state["final_message"] = last.content if hasattr(last, "content") else str(last)

    st.session_state["reject_mode"] = False
    st.session_state["phase"] = "complete"
    st.rerun()


# ── Phase: idle ───────────────────────────────────────────────────────────────

if st.session_state["phase"] == "idle":
    data_dir = Path("./data/samples")
    csvs = sorted(data_dir.glob("*.csv")) if data_dir.exists() else []
    options = [str(f) for f in csvs] or ["./data/samples/iris.csv"]

    col1, col2 = st.columns([3, 1])
    with col1:
        dataset_path = st.selectbox(
            "Select dataset",
            options=options,
            help="CSV file with a 'target' column",
        )
    with col2:
        run_button = st.button("▶ Run Pipeline", type="primary", use_container_width=True)

    if run_button:
        from mlops_agents.graphs.mlops_graph import graph
        from mlops_agents.config.constants import GRAPH_RECURSION_LIMIT

        thread_id = f"streamlit-{int(time.time())}"
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": GRAPH_RECURSION_LIMIT,
        }
        st.session_state["pipeline_config"] = config

        st.subheader("Pipeline Log")
        log_placeholder = st.empty()

        for event in graph.stream(build_initial_state(dataset_path), config=config):
            if "__interrupt__" in event:
                st.session_state["interrupt_value"] = event["__interrupt__"][0].value
                st.session_state["phase"] = "awaiting_approval"
                _log("⏸ **Pipeline paused — awaiting human approval**")
                _render_log(log_placeholder)
                st.rerun()
            else:
                line = event_to_log_line(event)
                if line:
                    _log(line)
                    _render_log(log_placeholder)

        # Stream ended without interrupt — pipeline finished inline
        if st.session_state["phase"] == "idle":
            final = graph.get_state(config).values
            st.session_state["deployment_decision"] = final.get("deployment_decision", "pending")
            msgs = final.get("messages", [])
            if msgs:
                last = msgs[-1]
                st.session_state["final_message"] = (
                    last.content if hasattr(last, "content") else str(last)
                )
            st.session_state["phase"] = "complete"
            st.rerun()


# ── Phase: awaiting_approval ──────────────────────────────────────────────────

elif st.session_state["phase"] == "awaiting_approval":
    st.subheader("Pipeline Log")
    st.markdown("\n\n".join(st.session_state["log_lines"]))

    st.divider()

    iv = st.session_state["interrupt_value"]
    st.subheader("⚠️ Human Approval Required")
    st.markdown(f"**{iv.get('question', 'Approve this action?')}**")

    summary = iv.get("registration_summary", "")
    if summary:
        with st.expander("Registration details"):
            st.text(summary)

    if not st.session_state["reject_mode"]:
        col_a, col_r = st.columns(2)
        with col_a:
            if st.button("✅ Approve", type="primary", use_container_width=True):
                _resume_pipeline({"approved": True})
        with col_r:
            if st.button("❌ Reject", use_container_width=True):
                st.session_state["reject_mode"] = True
                st.rerun()
    else:
        reason = st.text_input("Rejection reason (optional)")
        if st.button("Confirm Rejection", type="primary"):
            _resume_pipeline({"approved": False, "reason": reason or "Rejected by operator"})


# ── Phase: complete ───────────────────────────────────────────────────────────

elif st.session_state["phase"] == "complete":
    st.subheader("Pipeline Log")
    st.markdown("\n\n".join(st.session_state["log_lines"]))

    st.divider()

    decision = st.session_state["deployment_decision"]
    msg = st.session_state["final_message"]

    if decision == "approved":
        st.success("Pipeline complete. Model promoted to champion.")
    elif decision == "rejected":
        st.warning(f"Deployment rejected. {msg}" if msg else "Deployment rejected.")
    else:
        st.error(f"Pipeline stopped early. {msg}" if msg else "Pipeline stopped early.")

    if st.button("🔄 Run Again"):
        st.session_state.clear()
        st.rerun()
