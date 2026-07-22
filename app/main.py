"""FastAPI inference API for the quality inspection system."""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.config import OUTPUTS_DIR, UPLOADS_DIR
from app.detector import QualityInspector

app = FastAPI(
    title="Computer Vision Quality Inspection API",
    description="Upload an image and get object detections with confidence scores.",
    version=__version__,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once when the API starts (lazy on first request is also fine)
inspector: QualityInspector | None = None


def get_inspector() -> QualityInspector:
    global inspector
    if inspector is None:
        inspector = QualityInspector()
    return inspector


@app.get("/")
def root() -> dict:
    return {
        "message": "Computer Vision Quality Inspection API is running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    """Upload an image and get detections + quality grade."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    suffix = Path(file.filename or "upload.jpg").suffix or ".jpg"
    upload_name = f"{uuid.uuid4().hex}{suffix}"
    upload_path = UPLOADS_DIR / upload_name

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    upload_path.write_bytes(content)

    try:
        result = get_inspector().inspect(upload_path)
    except Exception as exc:  # noqa: BLE001 - return friendly API error
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc

    annotated_name = f"annotated_{upload_name}"
    annotated_path = OUTPUTS_DIR / annotated_name
    get_inspector().save_annotated_image(result["annotated_image"], annotated_path)

    return {
        "image_name": result["image_name"],
        "num_detections": result["num_detections"],
        "detections": result["detections"],
        "quality_grade": result["quality_grade"],
        "quality_score": result["quality_score"],
        "quality_notes": result["quality_notes"],
        "annotated_image_url": f"/outputs/{annotated_name}",
    }


@app.get("/outputs/{filename}")
def get_output_image(filename: str) -> FileResponse:
    """Download / view an annotated result image."""
    path = OUTPUTS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path)
