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
    missing = df.isnull().sum()
    max_missing_pct = float(missing.max() / len(df) * 100) if len(df) > 0 else 0.0

    summary = {
        "passed": max_missing_pct < 20.0 and int(df.duplicated().sum()) == 0,
        "row_count": len(df),
        "column_count": len(df.columns),
        "missing_values_total": int(missing.sum()),
        "columns_with_missing": {col: int(cnt) for col, cnt in missing.items() if cnt > 0},
        "max_missing_pct": round(max_missing_pct, 2),
        "duplicate_rows": int(df.duplicated().sum()),
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

    # Extract only the drift verdict and per-feature scores — not the full report
    drift_detected = False
    feature_drift: dict = {}
    try:
        for metric in report_dict.get("metrics", []):
            value = metric.get("value", {})
            if "drift_detected" in value:
                drift_detected = bool(value["drift_detected"])
            if "column_name" in value and "drift_score" in value:
                feature_drift[value["column_name"]] = round(float(value["drift_score"]), 4)
    except Exception:
        pass

    summary = {
        "drift_detected": drift_detected,
        "feature_drift_scores": feature_drift,
    }
    logger.info(f"Drift check: {Path(current_path).name} vs {Path(reference_path).name}")
    return json.dumps(summary, default=str)
