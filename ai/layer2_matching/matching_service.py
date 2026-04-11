from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from ai.layer2_matching.pipeline import Layer2Pipeline
from ai.layer2_matching.schemas import AnalysisResponse, SimilarContentItem


PROJECT_DIR = Path(__file__).resolve().parents[2]
pipeline = Layer2Pipeline(PROJECT_DIR)

app = FastAPI(
    title="Layer 2 Multimodal Similarity Tracking API",
    description="AI-Powered Deepfake Detection, Propagation Tracking, and Risk Analysis System - Layer 2",
    version="1.0.0",
)


@app.get("/")
def root() -> dict[str, object]:
    return {
        "status": "ok",
        "service": "layer2",
        "visual_index_size": pipeline.visual_index.size,
        "audio_index_size": pipeline.audio_index.size,
        "endpoints": ["/upload", "/similar/{id}", "/analysis/{id}"],
    }


@app.post("/upload", response_model=AnalysisResponse)
def upload_media(
    file: UploadFile = File(...),
    query_hint: str | None = Form(None),
    source_url: str | None = Form(None),
    is_fake: bool | None = Form(None),
    confidence: float | None = Form(None),
) -> AnalysisResponse:
    suffix = Path(file.filename or "upload.bin").suffix or ".bin"
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = Path(temp_file.name)
        return pipeline.analyze_media(
            source_path=temp_path,
            original_filename=file.filename,
            query_hint=query_hint,
            source_url=source_url,
            is_fake=is_fake,
            confidence=confidence,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - API safety
        raise HTTPException(status_code=500, detail=f"Layer 2 analysis failed: {exc}") from exc
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


@app.get("/similar/{analysis_id}", response_model=list[SimilarContentItem])
def get_similar(analysis_id: str) -> list[SimilarContentItem]:
    try:
        return pipeline.load_similar(analysis_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/analysis/{analysis_id}", response_model=AnalysisResponse)
def get_analysis(analysis_id: str) -> AnalysisResponse:
    try:
        return pipeline.load_analysis(analysis_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
