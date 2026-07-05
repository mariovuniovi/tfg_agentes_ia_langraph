"""Typed wrapper over MLflow's MlflowClient — returns Pydantic models shaped for charts."""
from datetime import UTC, datetime
from typing import Literal

import mlflow
from mlflow.tracking import MlflowClient

from api.models.experiment import ExperimentOut, MetricSeries, RunOut
from mlops_agents.config.settings import settings

_LINE_STYLES: list[Literal["solid", "dashed", "dotted"]] = ["solid", "dashed", "dotted"]


class MlflowService:
    def __init__(self) -> None:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        self._client = MlflowClient()

    def list_experiments(self) -> list[ExperimentOut]:
        exps = self._client.search_experiments()
        return [
            ExperimentOut(experiment_id=e.experiment_id, name=e.name)
            for e in exps
        ]

    def get_runs(self, experiment_id: str, max_results: int = 50) -> list[RunOut]:
        runs = self._client.search_runs(
            experiment_ids=[experiment_id],
            order_by=["start_time DESC"],
            max_results=max_results,
        )
        return [self._to_run_out(r) for r in runs]

    def _to_run_out(self, run) -> RunOut:
        metrics: dict[str, float] = dict(run.data.metrics)

        metric_series: list[MetricSeries] = []
        for idx, metric_name in enumerate(sorted(metrics.keys())):
            history = self._client.get_metric_history(run.info.run_id, metric_name)
            if history:
                metric_series.append(MetricSeries(
                    name=metric_name,
                    steps=[m.step for m in history],
                    values=[m.value for m in history],
                    line_style=_LINE_STYLES[idx % len(_LINE_STYLES)],
                ))

        return RunOut(
            run_id=run.info.run_id,
            run_name=run.info.run_name or run.info.run_id[:8],
            status=run.info.status,
            start_time=datetime.fromtimestamp(
                run.info.start_time / 1000, tz=UTC
            ),
            params=dict(run.data.params),
            metrics=metrics,
            metric_series=metric_series,
        )
