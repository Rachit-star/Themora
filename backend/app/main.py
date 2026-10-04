"""
Themora API -- upload a CSV, get data-quality findings back.
"""

import os
import tempfile

import pandas as pd
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.detectors import run_all

app = FastAPI(title="Themora", version="0.1.0")

MAX_FILE_SIZE = 200 * 1024 * 1024  # 200 MB
CHUNK_SIZE = 1024 * 1024           # 1 MB read chunks

# Allow the Next.js frontend (dev server) to call us
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -- Health check --------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


# -- Audit endpoint ------------------------------------------------------------

@app.post("/audit")
async def audit(
    file: UploadFile = File(...),
    target: str | None = Form(default=None),
):
    """
    Upload a CSV file and receive a list of data-quality findings.

    - **file**: CSV file to audit (max 200 MB)
    - **target**: (optional) name of the target / label column
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are supported.")

    # Stream the upload to a temp file so we never hold the raw bytes
    # and the parsed DataFrame in memory at the same time.
    tmp_path = None
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".csv")
        size = 0
        with os.fdopen(tmp_fd, "wb") as tmp:
            while chunk := await file.read(CHUNK_SIZE):
                size += len(chunk)
                if size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)} MB.",
                    )
                tmp.write(chunk)

        try:
            df = pd.read_csv(tmp_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")
    finally:
        # Always clean up the temp file
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if df.empty:
        return {"findings": [], "summary": _summary([], df)}

    if target and target not in df.columns:
        raise HTTPException(
            status_code=400,
            detail=f"Target column '{target}' not found. Columns: {list(df.columns)}",
        )

    findings = run_all(df, target=target)
    return {
        "findings": [f.to_dict() for f in findings],
        "summary": _summary(findings, df),
    }


# -- Summary helper ------------------------------------------------------------

def _summary(findings, df=None):
    """Quick counts by severity + basic dataset info for the frontend."""
    counts = {"high": 0, "medium": 0, "low": 0, "total": len(findings)}
    for f in findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1
    result = {"counts": counts}
    if df is not None:
        result["dataset"] = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": list(df.columns),
        }
    return result
