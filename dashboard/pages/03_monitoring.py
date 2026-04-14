"""Monitoring page — data drift and model performance overview."""

import streamlit as st

st.set_page_config(page_title="Monitoring", layout="wide")
st.title("Model Monitoring")

st.info("Upload a current dataset to compare against the reference for drift detection.")

col1, col2 = st.columns(2)
with col1:
    reference_file = st.file_uploader("Reference dataset (CSV)", type="csv", key="ref")
with col2:
    current_file = st.file_uploader("Current dataset (CSV)", type="csv", key="cur")

if reference_file and current_file and st.button("Run Drift Detection"):
    import tempfile
    from pathlib import Path

    import pandas as pd

    with tempfile.TemporaryDirectory() as tmp:
        ref_path = Path(tmp) / "reference.csv"
        cur_path = Path(tmp) / "current.csv"
        ref_path.write_bytes(reference_file.read())
        cur_path.write_bytes(current_file.read())

        try:
            from evidently import Report
            from evidently.presets import DataDriftPreset

            reference = pd.read_csv(ref_path)
            current = pd.read_csv(cur_path)

            report = Report([DataDriftPreset(method="psi")])
            result = report.run(reference, current)
            report_dict = result.as_dict()

            st.subheader("Drift Report")
            st.json(report_dict)

        except Exception as e:
            st.error(f"Drift detection failed: {e}")
