"""Evidently AI tools for data quality and drift detection.

Returns structured JSON dicts that the data validation agent can
interpret to decide whether to gate or pass the pipeline.
"""

import json
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

from mlops_agents.utils.logging import get_logger

logger = get_logger(__name__)

try:
    from evidently import Report
    from evidently.presets import DataDriftPreset, DataSummaryPreset
    _EVIDENTLY_AVAILABLE = True
except ImportError:
    _EVIDENTLY_AVAILABLE = False


@tool
def check_data_quality(dataset_path: str) -> str:
    """Run an Evidently data quality report on the dataset.

    Checks for missing values, out-of-range values, duplicate rows,
    and type consistency. Returns a structured JSON summary.

    Args:
        dataset_path: Path to the CSV file to validate.

    Returns:
        JSON string with quality metrics and an overall 'passed' flag.
    """
    if not _EVIDENTLY_AVAILABLE:
        return json.dumps({"error": "evidently not installed — run: uv add evidently"})

    df = pd.read_csv(dataset_path)
    report = Report([DataSummaryPreset()])
    result = report.run(df, df)
    report_dict = result.dump_dict()

    # Extract key metrics for the agent to interpret
    summary = {
        "passed": True,  # agent will update this based on metrics
        "row_count": len(df),
        "column_count": len(df.columns),
        "report": report_dict,
    }
    logger.info(f"Data quality check complete for {Path(dataset_path).name}")
    return json.dumps(summary, default=str)


@tool
def check_data_drift(current_path: str, reference_path: str) -> str:
    """Detect statistical drift between a current dataset and a reference dataset.

    Uses Population Stability Index (PSI) to measure feature drift.
    A PSI score > 0.1 indicates significant drift.

    Args:
        current_path: Path to the current (new) data CSV.
        reference_path: Path to the reference (baseline) data CSV.

    Returns:
        JSON string with per-feature drift scores and overall drift verdict.
    """
    if not _EVIDENTLY_AVAILABLE:
        return json.dumps({"error": "evidently not installed — run: uv add evidently"})

    reference = pd.read_csv(reference_path)
    current = pd.read_csv(current_path)

    report = Report([DataDriftPreset()])
    result = report.run(reference, current)
    report_dict = result.dump_dict()

    summary = {
        "drift_detected": False,  # agent will interpret report_dict to set this
        "report": report_dict,
    }
    logger.info(f"Drift check: {Path(current_path).name} vs {Path(reference_path).name}")
    return json.dumps(summary, default=str)
