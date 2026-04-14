"""Experiments page — browse MLflow runs."""

import streamlit as st

st.set_page_config(page_title="Experiments", layout="wide")
st.title("MLflow Experiments")

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    from mlops_agents.config.settings import settings

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()

    experiments = client.search_experiments()
    exp_names = [e.name for e in experiments]

    if not exp_names:
        st.info("No experiments found. Run the pipeline first.")
    else:
        selected_exp = st.selectbox("Experiment", exp_names)
        exp = client.get_experiment_by_name(selected_exp)
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=20,
        )

        if runs:
            import pandas as pd
            rows = [
                {"run_id": r.info.run_id[:8], **r.data.metrics, **r.data.params}
                for r in runs
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("No runs yet for this experiment.")

except Exception as e:
    st.error(f"Could not connect to MLflow: {e}")
    st.info(f"Make sure MLflow is running at the configured URI.")
