"""Experiments router: MLflow experiment and run data."""
from fastapi import APIRouter, HTTPException

from api.models.experiment import ExperimentOut, RunOut
from api.services.mlflow_client import MlflowService

router = APIRouter()


@router.get("/experiments", response_model=list[ExperimentOut])
async def list_experiments():
    try:
        svc = MlflowService()
        return svc.list_experiments()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"MLflow unavailable: {exc}") from exc


@router.get("/experiments/{experiment_id}/runs", response_model=list[RunOut])
async def get_experiment_runs(experiment_id: str):
    try:
        svc = MlflowService()
        return svc.get_runs(experiment_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"MLflow unavailable: {exc}") from exc
