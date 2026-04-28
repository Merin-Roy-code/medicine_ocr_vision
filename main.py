"""
main.py — FastAPI application entry point.

Endpoints:
  POST /extract   — Upload a medicine strip image; receive structured JSON.
  GET  /health    — Liveness check.
  GET  /docs      — Swagger UI (built-in FastAPI).

Run with:
  uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import io
import sys
import time
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field

import config
from extractor import extract_all
from fuzzy_match import run_fuzzy_matching
from ocr import run_ocr
from preprocess import preprocess

# ──────────────────────────────────────────────────────────────────────────────
# Logging configuration
# ──────────────────────────────────────────────────────────────────────────────

logger.remove()   # Remove default handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>",
    level="DEBUG",
    colorize=True,
)
logger.add(
    "logs/medicine_ocr.log",
    rotation="10 MB",
    retention="7 days",
    level="INFO",
    format="{time} | {level} | {name}:{line} — {message}",
)

# ──────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ──────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Medicine Strip OCR API",
    description=(
        "Extracts structured information (medicine name, batch, expiry, "
        "manufacturer) from medicine strip images using Google Cloud Vision OCR "
        "and RapidFuzz matching."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow all origins in development; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────────────────
# Pydantic response schema
# ──────────────────────────────────────────────────────────────────────────────

class ExtractionResponse(BaseModel):
    """JSON response returned by POST /extract."""

    medicine_name: Optional[str] = Field(
        None, description="Best-matched canonical medicine name from the database."
    )
    confidence: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Confidence score for the medicine name match (0.0–1.0)."
    )
    batch: Optional[str] = Field(
        None, description="Extracted batch / lot number."
    )
    expiry: Optional[str] = Field(
        None, description="Normalised expiry date in MM/YYYY format."
    )
    manufacturer: Optional[str] = Field(
        None, description="Extracted and fuzzy-corrected manufacturer name."
    )
    composition: Optional[str] = Field(
        None, description="Extracted chemical composition / active ingredients."
    )
    raw_ocr_text: str = Field(
        "", description="Full raw text returned by Google Cloud Vision."
    )
    processing_time_ms: float = Field(
        0.0, description="Total server-side processing time in milliseconds."
    )

    model_config = {"json_schema_extra": {
        "example": {
            "medicine_name": "Norfloxacin 400",
            "confidence": 0.93,
            "batch": "NF1234",
            "expiry": "05/2026",
            "manufacturer": "Cipla Ltd",
            "raw_ocr_text": "NORFLAXEM 400MG …",
            "processing_time_ms": 1842.5,
        }
    }}


class HealthResponse(BaseModel):
    status: str
    version: str


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline orchestrator (called by the endpoint)
# ──────────────────────────────────────────────────────────────────────────────

def run_pipeline(image_bytes: bytes) -> ExtractionResponse:
    """
    Execute the full OCR pipeline for one image.

    Stages:
      1. Preprocess  — OpenCV enhancement (CLAHE, denoise, sharpen, normalize).
      2. OCR         — Google Cloud Vision DOCUMENT_TEXT_DETECTION.
      3. Extract     — Text cleaning + regex field extraction.
      4. Fuzzy match — RapidFuzz correction of medicine name + manufacturer.
      5. Assemble    — Build the final ExtractionResponse.

    Args:
        image_bytes: Raw bytes of the uploaded image file.

    Returns:
        ExtractionResponse with all extracted fields.
    """
    t_start = time.perf_counter()

    # ── Stage 1: Preprocess ────────────────────────────────────────────────
    logger.info("Stage 1/4 — Preprocessing image …")
    _, preprocessed_bytes = preprocess(image_bytes)

    # ── Stage 2: OCR ───────────────────────────────────────────────────────
    logger.info("Stage 2/4 — Running OCR …")
    full_text, blocks = run_ocr(preprocessed_bytes)
    block_texts = [b.text for b in blocks]

    # ── Stage 3: Extract ───────────────────────────────────────────────────
    logger.info("Stage 3/4 — Extracting fields …")
    raw = extract_all(full_text, block_texts)

    # ── Stage 4: Fuzzy match ───────────────────────────────────────────────
    logger.info("Stage 4/4 — Running fuzzy matching …")
    med_result, mfr_result = run_fuzzy_matching(
        medicine_candidates=raw.medicine_name_candidates,
        raw_manufacturer=raw.manufacturer,
    )

    # ── Assemble response ──────────────────────────────────────────────────
    elapsed_ms = (time.perf_counter() - t_start) * 1000

    # If fuzzy match returned None (no DB match), fall back to the best raw
    # candidate if available, and set a lower confidence.
    final_name = med_result.matched_value
    final_confidence = med_result.confidence

    if final_name is None and raw.medicine_name_candidates:
        final_name = raw.medicine_name_candidates[0].title()
        final_confidence = 0.30   # low confidence — regex-only result

    response = ExtractionResponse(
        medicine_name=final_name,
        confidence=round(final_confidence, 4),
        batch=raw.batch,
        expiry=raw.expiry,
        manufacturer=mfr_result.matched_value,
        composition=raw.composition,
        raw_ocr_text=full_text,
        processing_time_ms=round(elapsed_ms, 2),
    )

    logger.info(
        f"Pipeline complete in {elapsed_ms:.1f} ms | "
        f"name={response.medicine_name} | batch={response.batch} | "
        f"expiry={response.expiry} | mfr={response.manufacturer}"
    )
    return response


# ──────────────────────────────────────────────────────────────────────────────
# API Endpoints
# ──────────────────────────────────────────────────────────────────────────────

ALLOWED_CONTENT_TYPES = {
    "image/jpeg", "image/jpg", "image/png",
    "image/webp", "image/bmp", "image/tiff",
}


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Liveness check",
)
async def health() -> HealthResponse:
    """Returns 200 OK if the service is running."""
    return HealthResponse(status="ok", version="1.0.0")


@app.post(
    "/extract",
    response_model=ExtractionResponse,
    tags=["OCR"],
    summary="Extract medicine information from a strip image",
    status_code=status.HTTP_200_OK,
)
async def extract(
    image: UploadFile = File(
        ...,
        description="Medicine strip image (JPEG, PNG, WEBP, BMP, or TIFF).",
    ),
) -> ExtractionResponse:
    """
    Upload a medicine strip photo and receive structured extraction results.

    **Pipeline stages:**
    1. OpenCV preprocessing (CLAHE, denoising, sharpening, glare reduction)
    2. Google Cloud Vision OCR (DOCUMENT_TEXT_DETECTION)
    3. Regex-based field extraction with OCR error correction
    4. RapidFuzz medicine name & manufacturer fuzzy matching

    **Returns:** JSON with `medicine_name`, `confidence`, `batch`, `expiry`,
    `manufacturer`, `raw_ocr_text`, and `processing_time_ms`.
    """
    # ── Validate file type ─────────────────────────────────────────────────
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type: '{image.content_type}'. "
                f"Allowed types: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}"
            ),
        )

    # ── Validate file size (max 15 MB) ─────────────────────────────────────
    MAX_SIZE_BYTES = 15 * 1024 * 1024
    image_bytes = await image.read()
    if len(image_bytes) > MAX_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image too large ({len(image_bytes) / 1024 / 1024:.1f} MB). Maximum is 15 MB.",
        )

    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    logger.info(
        f"Received image: '{image.filename}' | "
        f"type={image.content_type} | size={len(image_bytes) / 1024:.1f} KB"
    )

    # ── Run pipeline ───────────────────────────────────────────────────────
    try:
        result = run_pipeline(image_bytes)
    except EnvironmentError as exc:
        # Missing credentials — surface as 503 (server configuration error)
        logger.error(f"EnvironmentError: {exc}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )
    except RuntimeError as exc:
        logger.error(f"Pipeline error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception(f"Unexpected error during pipeline: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Check server logs.",
        )

    return result


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.makedirs("logs", exist_ok=True)
    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True,
        log_level="info",
    )
