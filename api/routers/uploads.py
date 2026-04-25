"""Uploads router: receive CSV files, store them, return server paths."""
import os
import uuid

from fastapi import APIRouter, HTTPException, UploadFile
from typing import Optional

UPLOAD_DIR = "data/uploads"

router = APIRouter()


@router.post("/uploads")
async def upload_files(files: Optional[list[UploadFile]] = None) -> dict:
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
