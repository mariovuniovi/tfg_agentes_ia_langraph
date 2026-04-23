from datetime import datetime
from pydantic import BaseModel


class ColumnDriftResult(BaseModel):
    column: str
    drift_detected: bool
    score: float
    method: str


class DriftReport(BaseModel):
    dataset_drift: bool
    drift_share: float
    columns: list[ColumnDriftResult]
    generated_at: datetime
