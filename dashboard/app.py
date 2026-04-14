"""Streamlit multi-page dashboard entry point.

Run with:
    uv run streamlit run dashboard/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="MLOps Multi-Agent System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("MLOps Multi-Agent System")
st.markdown(
    """
    Welcome to the MLOps pipeline dashboard.
    Use the sidebar to navigate between pages:

    - **Pipeline** — run the full agent pipeline on a dataset
    - **Experiments** — browse MLflow experiment runs
    - **Monitoring** — data drift and model performance
    - **Chat** — interact with agents via natural language
    """
)

st.info("Select a page from the sidebar to get started.")
