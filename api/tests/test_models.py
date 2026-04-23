"""Tests for Pydantic API models."""
from datetime import datetime
from api.models.run import RunCreate, RunStatus, PipelineEventModel, HITLDecision
from api.models.experiment import MetricSeries, RunOut, ExperimentOut
from api.models.monitoring import ColumnDriftResult, DriftReport


def test_run_create_requires_dataset_paths():
    rc = RunCreate(dataset_paths=["data/samples/iris_measurements.csv"])
    assert rc.dataset_paths == ["data/samples/iris_measurements.csv"]


def test_hitl_decision_valid_values():
    d = HITLDecision(decision="approve")
    assert d.decision == "approve"
    d2 = HITLDecision(decision="reject", reason="too low accuracy")
    assert d2.reason == "too low accuracy"


def test_hitl_decision_rejects_invalid():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        HITLDecision(decision="maybe")


def test_run_status_model():
    rs = RunStatus(run_id="abc", status="running")
    assert rs.interrupt_value is None
    rs2 = RunStatus(run_id="abc", status="awaiting_approval", interrupt_value={"model": "v1"})
    assert rs2.interrupt_value == {"model": "v1"}


def test_pipeline_event_model():
    ev = PipelineEventModel(
        type="tool_call",
        agent="trainer",
        timestamp_ms=1000.0,
        data={"tool_name": "train_model", "arguments": {}},
    )
    assert ev.type == "tool_call"


def test_metric_series_line_styles():
    ms = MetricSeries(name="accuracy", steps=[1, 2], values=[0.8, 0.9], line_style="solid")
    assert ms.line_style == "solid"


def test_drift_report_model():
    col = ColumnDriftResult(column="feature_1", drift_detected=True, score=0.3, method="psi")
    dr = DriftReport(
        dataset_drift=True,
        drift_share=0.5,
        columns=[col],
        generated_at=datetime.utcnow(),
    )
    assert dr.drift_share == 0.5
    assert dr.columns[0].column == "feature_1"
