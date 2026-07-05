"""Uploads router: receive CSV files, store them, return server paths."""
import json
import os
import uuid

from fastapi import APIRouter, HTTPException, UploadFile
from pydantic import ValidationError

from mlops_agents.contracts.schema import SchemaContract

UPLOAD_DIR = "data/uploads"

router = APIRouter()


@router.post("/uploads")
async def upload_files(
    # Optional so handler can return 400 — required list[UploadFile] would raise FastAPI's own 422 first
    files: list[UploadFile] | None = None,
) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    for f in files:
        filename = f.filename or ""
        if not filename.lower().endswith(".csv"):
            raise HTTPException(
                status_code=422,
                detail=f"Only CSV files are accepted, got: {filename}",
            )

    os.makedirs(UPLOAD_DIR, exist_ok=True)

    saved_paths: list[str] = []
    for f in files:
        prefix = uuid.uuid4().hex[:8]
        dest = os.path.join(UPLOAD_DIR, f"{prefix}_{f.filename}")
        content = await f.read()
        with open(dest, "wb") as fh:
            fh.write(content)
        saved_paths.append(dest)

    return {"paths": saved_paths}


@router.post("/uploads/schema")
async def validate_schema(file: UploadFile) -> dict:
    """Validate a schema JSON file against SchemaContract. Returns the raw JSON on success."""
    raw = await file.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"Not valid JSON: {exc}") from exc
    try:
        SchemaContract.model_validate(data)
    except ValidationError as exc:
        first = exc.errors()[0]["msg"]
        raise HTTPException(status_code=422, detail=f"Schema contract violation: {first}") from exc
    return {"schema_json": raw.decode("utf-8"), "problem_type": data["problem_type"]}
