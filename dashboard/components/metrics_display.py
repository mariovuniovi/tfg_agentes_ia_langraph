"""Reusable metrics display component."""

import streamlit as st


def show_metrics(metrics: dict, title: str = "Metrics") -> None:
    """Display a dict of metrics as Streamlit metric cards."""
    st.subheader(title)
    cols = st.columns(len(metrics))
    for col, (key, value) in zip(cols, metrics.items()):
        with col:
            if isinstance(value, float):
                st.metric(key, f"{value:.4f}")
            else:
                st.metric(key, str(value))
