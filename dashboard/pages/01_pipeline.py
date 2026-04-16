"""Pipeline page — run and monitor the MLOps agent pipeline.

Three-phase state machine stored in st.session_state:
  idle             → dataset selector + Run button
  awaiting_approval → full-width frozen log + HITL approval panel
  complete         → two-column log (left) + results tabs (right) + outcome banner
"""

import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st
from langgraph.types import Command

sys.path.insert(0, str(Path(__file__).parents[2]))

from dashboard.pipeline_helpers import build_initial_state, event_to_log_line, extract_panel_data
from mlops_agents.config.constants import GRAPH_RECURSION_LIMIT
from mlops_agents.graphs.mlops_graph import graph

st.set_page_config(page_title="Pipeline", layout="wide")
st.title("🤖 MLOps Pipeline")

# ── Session state initialisation ──────────────────────────────────────────────
_DEFAULTS: dict = {
    "phase": "idle",  # idle | awaiting_approval | complete
    "log_lines": [],
    "pipeline_config": None,
    "interrupt_value": {},
    "deployment_decision": "pending",
    "final_message": "",
    "reject_mode": False,
    "validation_report": {},
    "training_metrics": {},
    "evaluation_report": {},
    "dataset_preview": [],
    "training_run_id": "",
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Helpers ───────────────────────────────────────────────────────────────────


def _log(line: str) -> None:
    st.session_state["log_lines"].append(line)


def _render_log(placeholder) -> None:
    placeholder.markdown("\n\n".join(st.session_state["log_lines"]))


def _update_panel_data(config: dict) -> None:
    """Read current graph state and update panel session fields."""
    state_vals = graph.get_state(config).values
    panel = extract_panel_data(state_vals)
    for key, val in panel.items():
        if val:  # never overwrite populated data with empty values from intermediate state polls
            st.session_state[key] = val
    run_id = state_vals.get("training_run_id", "")
    if run_id:
        st.session_state["training_run_id"] = run_id


def _render_tabs(right_placeholder) -> None:
    """Render the three results tabs into right_placeholder."""
    with right_placeholder.container():
        tab1, tab2, tab3 = st.tabs(["📊 Data", "🏋️ Training", "📈 Evaluation"])

        with tab1:
            report = st.session_state["validation_report"]
            if not report:
                st.info("Waiting for data validation to complete...")
            else:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Rows", report.get("row_count", "—"))
                c2.metric("Columns", report.get("column_count", "—"))
                c3.metric("Missing %", f"{report.get('max_missing_pct', 0):.1f}%")
                c4.metric("Status", "✅ Passed" if report.get("passed") else "❌ Failed")
                preview = st.session_state["dataset_preview"]
                if preview:
                    st.dataframe(pd.DataFrame(preview), use_container_width=True)
                with st.expander("Full Validation Report"):
                    st.json(report)

        with tab2:
            metrics = st.session_state["training_metrics"]
            run_id = st.session_state["training_run_id"]
            if not metrics:
                st.info("Waiting for model training to complete...")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Model", metrics.get("model_type", "—"))
                c2.metric("Train Acc", f"{metrics.get('train_accuracy', 0):.2%}")
                c3.metric("Val Acc", f"{metrics.get('val_accuracy', 0):.2%}")
                if run_id:
                    st.caption(f"MLflow Run ID: `{run_id}`")
                num_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
                if num_metrics:
                    st.dataframe(
                        pd.DataFrame([num_metrics]).T.rename(columns={0: "Value"}),
                        use_container_width=True,
                    )

        with tab3:
            eval_report = st.session_state["evaluation_report"]
            if not eval_report:
                st.info("Waiting for model evaluation to complete...")
            else:
                candidate = eval_report.get("candidate_metrics", {})
                baseline = eval_report.get("baseline_metrics", {})
                if candidate or baseline:
                    all_keys = sorted(set(list(candidate.keys()) + list(baseline.keys())))
                    rows = [
                        {
                            "Metric": k,
                            "Candidate": candidate.get(k, "—"),
                            "Baseline": baseline.get(k, "—"),
                        }
                        for k in all_keys
                    ]
                    st.dataframe(pd.DataFrame(rows).set_index("Metric"), use_container_width=True)
                cand_run = eval_report.get("candidate_run_id", "")
                if cand_run:
                    st.caption(f"Candidate Run ID: `{cand_run}`")
                if not candidate and not baseline:
                    st.json(eval_report)


def _resume_pipeline(resume: dict) -> None:
    """Resume the paused graph with the operator decision and stream remaining events."""
    config = st.session_state["pipeline_config"]
    log_placeholder = st.empty()
    _render_log(log_placeholder)

    for event in graph.stream(Command(resume=resume), config=config):
        line = event_to_log_line(event)
        if line:
            _log(line)
            _render_log(log_placeholder)

    _update_panel_data(config)

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
    from mlops_agents.config.settings import settings

    data_dir = Path(settings.data_dir)
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
        thread_id = f"streamlit-{int(time.time())}"
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": GRAPH_RECURSION_LIMIT,
        }
        st.session_state["pipeline_config"] = config

        left_col, right_col = st.columns([4, 6])
        with left_col:
            st.subheader("Pipeline Log")
            log_placeholder = st.empty()
        with right_col:
            st.subheader("Live Results")
            right_placeholder = st.empty()

        interrupt_detected = False
        for event in graph.stream(build_initial_state(dataset_path), config=config):
            if "__interrupt__" in event:
                st.session_state["interrupt_value"] = event["__interrupt__"][0].value
                st.session_state["phase"] = "awaiting_approval"
                _log("⏸ **Pipeline paused — awaiting human approval**")
                interrupt_detected = True
                break
            else:
                line = event_to_log_line(event)
                if line:
                    _log(line)
                    _render_log(log_placeholder)
                node = next(iter(event), None)
                if node in ("data_validator", "trainer", "evaluator"):
                    _update_panel_data(config)
                    _render_tabs(right_placeholder)

        if interrupt_detected:
            _render_log(log_placeholder)
            st.rerun()
        elif st.session_state["phase"] == "idle":
            _update_panel_data(config)
            final = graph.get_state(config).values
            st.session_state["deployment_decision"] = final.get("deployment_decision", "pending")
            msgs = final.get("messages", [])
            if msgs:
                last = msgs[-1]
                st.session_state["final_message"] = last.content if hasattr(last, "content") else str(last)
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
    left_col, right_col = st.columns([4, 6])
    with left_col:
        st.subheader("Pipeline Log")
        st.markdown("\n\n".join(st.session_state["log_lines"]))
    with right_col:
        st.subheader("Results")
        right_placeholder = st.empty()
        _render_tabs(right_placeholder)

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
        for k, v in _DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()
