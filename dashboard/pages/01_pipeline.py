"""Pipeline page — trigger and monitor the MLOps agent pipeline."""

from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="Pipeline", layout="wide")
st.title("MLOps Pipeline")

# Dataset selection
data_dir = Path("./data/samples")
datasets = list(data_dir.glob("*.csv")) if data_dir.exists() else []
dataset_options = [str(f) for f in datasets]

col1, col2 = st.columns([3, 1])
with col1:
    dataset_path = st.selectbox(
        "Select dataset",
        options=dataset_options or ["./data/samples/iris.csv"],
        help="CSV file with a 'target' column",
    )
with col2:
    run_button = st.button("Run Pipeline", type="primary", use_container_width=True)

st.divider()

# Pipeline execution
if run_button:
    from mlops_agents.graphs.mlops_graph import graph
    from mlops_agents.config.constants import GRAPH_RECURSION_LIMIT

    config = {
        "configurable": {"thread_id": f"streamlit-{st.session_state.get('run_count', 0)}"},
        "recursion_limit": GRAPH_RECURSION_LIMIT,
    }
    st.session_state["run_count"] = st.session_state.get("run_count", 0) + 1

    initial_state = {
        "messages": [HumanMessage(content=f"Run the full MLOps pipeline on dataset: {dataset_path}")],
        "next": "",
        "dataset_path": dataset_path,
        "validation_passed": False,
        "validation_report": {},
        "trained_model_path": "",
        "training_run_id": "",
        "training_metrics": {},
        "evaluation_passed": False,
        "evaluation_report": {},
        "best_model_uri": "",
        "deployment_decision": "pending",
        "deployment_status": "",
        "error_message": "",
        "retry_count": 0,
    }

    log_container = st.container()
    with log_container:
        st.subheader("Pipeline Log")
        log_placeholder = st.empty()
        log_lines: list[str] = []

        for event in graph.stream(initial_state, config=config):
            for node_name, _ in event.items():
                log_lines.append(f"✅ `{node_name}` completed")
                log_placeholder.markdown("\n".join(log_lines))

            # Handle HITL interrupt
            state_snapshot = graph.get_state(config)
            if state_snapshot.next and "__interrupt__" in str(state_snapshot.values):
                st.warning("Pipeline paused — human approval required.")
                st.session_state["pending_config"] = config
                break

    st.success("Pipeline run complete.")
