from datetime import datetime
from typing import Literal
from pydantic import BaseModel


class MetricSeries(BaseModel):
    name: str
    steps: list[int]
    values: list[float]
    line_style: Literal["solid", "dashed", "dotted"]


class RunOut(BaseModel):
    run_id: str
    run_name: str
    status: str
    start_time: datetime
    params: dict[str, str]
    metrics: dict[str, float]
    metric_series: list[MetricSeries]


class ExperimentOut(BaseModel):
    experiment_id: str
    name: str
