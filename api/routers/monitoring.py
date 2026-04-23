"""Monitoring router: automatic drift report + ad-hoc multi-file drift detection."""
import io
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, Form, HTTPException, UploadFile

import api.services.run_store as run_store
from api.services.pipeline import run_evidently

router = APIRouter()


@router.get("/monitoring/latest")
async def get_latest_drift():
    report = run_store.get_latest_drift_report()
    if report is None:
        raise HTTPException(status_code=404, detail="No completed pipeline run yet")
    return report


@router.post("/monitoring/drift")
async def drift_compare(
    files: list[UploadFile],
    reference_index: Annotated[int, Form()],
    current_index: Annotated[int, Form()],
):
    if reference_index >= len(files) or current_index >= len(files):
        raise HTTPException(
            status_code=400,
            detail=f"Index out of range: got {len(files)} files, "
                   f"reference_index={reference_index}, current_index={current_index}",
        )
    try:
        ref_bytes = await files[reference_index].read()
        cur_bytes = await files[current_index].read()
        reference_df = pd.read_csv(io.BytesIO(ref_bytes))
        current_df = pd.read_csv(io.BytesIO(cur_bytes))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {exc}")

    try:
        report = run_evidently(reference_df, current_df)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Drift detection failed: {exc}")

    return report
