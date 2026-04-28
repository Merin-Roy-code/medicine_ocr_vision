"""
ocr.py — Gemini Vision OCR integration.

Responsibilities:
  1. Send preprocessed image bytes to Gemini Vision for full-text extraction.
  2. Return both the full-text string AND a list of TextBlock objects,
     maintaining the same interface as the original Cloud Vision implementation
     so the rest of the pipeline (extractor.py, main.py) needs no changes.

Note:
  Gemini returns a single block of text rather than spatially separated blocks.
  The TextBlock list will contain one entry with the full text, which is
  sufficient for the downstream regex-based extractor.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import google.generativeai as genai
from loguru import logger

import config


# ──────────────────────────────────────────────────────────────────────────────
# Data model (unchanged — keeps interface compatible with extractor.py)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TextBlock:
    """Represents a single OCR text block."""
    text: str
    confidence: float
    # Bounding box vertices — not provided by Gemini; kept for interface compat.
    vertices: list[tuple[int, int]] = field(default_factory=list)

    @property
    def top_left_y(self) -> int:
        return min(v[1] for v in self.vertices) if self.vertices else 0

    @property
    def top_left_x(self) -> int:
        return min(v[0] for v in self.vertices) if self.vertices else 0


# ──────────────────────────────────────────────────────────────────────────────
# Gemini client setup
# ──────────────────────────────────────────────────────────────────────────────

_GEMINI_MODEL = "gemini-flash-latest"

# Prompt instructs Gemini to act as a pure OCR engine — return only the text,
# nothing else, so the existing extractor.py regex logic works unchanged.
_OCR_PROMPT = (
    "You are an OCR engine. Look at this medicine strip image carefully and "
    "extract EVERY piece of text you can see, exactly as printed. "
    "Include: medicine name, batch number, expiry date, manufacturer, "
    "dosage, composition, and any other text. "
    "Return ONLY the raw extracted text, line by line. "
    "Do NOT add explanations, labels, or formatting."
)


def _build_gemini_client() -> genai.GenerativeModel:
    """
    Configure and return a Gemini GenerativeModel instance.

    Raises:
        EnvironmentError: If GEMINI_API_KEY is not set.
    """
    api_key = config.GEMINI_API_KEY
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set. "
            "Add it to your .env file: GEMINI_API_KEY=AIzaSy..."
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(_GEMINI_MODEL)


# ──────────────────────────────────────────────────────────────────────────────
# Public OCR function
# ──────────────────────────────────────────────────────────────────────────────

def run_ocr(image_bytes: bytes) -> tuple[str, list[TextBlock]]:
    """
    Send image bytes to Gemini Vision and return structured OCR output.

    Args:
        image_bytes: PNG/JPEG bytes of the (preprocessed) image.

    Returns:
        full_text:  The complete text extracted from the image.
        blocks:     List containing a single TextBlock with the full text.
                    (Gemini does not return spatial block data.)

    Raises:
        EnvironmentError: If GEMINI_API_KEY is not configured.
        RuntimeError:     On any Gemini API failure.
    """
    model = _build_gemini_client()

    # Gemini accepts image bytes via the 'inline_data' part
    image_part = {
        "mime_type": "image/jpeg",
        "data": image_bytes,
    }

    logger.info(f"Calling Gemini Vision ({_GEMINI_MODEL}) for OCR …")

    try:
        response = model.generate_content([_OCR_PROMPT, image_part])
        full_text: str = response.text.strip() if response.text else ""
    except Exception as exc:
        raise RuntimeError(f"Gemini Vision API error: {exc}") from exc

    if not full_text:
        logger.warning("Gemini returned empty text for this image.")
        return "", []

    logger.debug(f"Gemini OCR output ({len(full_text)} chars):\n{full_text[:300]}")

    # Wrap the result in a single TextBlock to satisfy the pipeline interface
    blocks = [TextBlock(text=full_text, confidence=0.95)]

    logger.info(f"Gemini OCR complete — extracted {len(full_text)} characters.")
    return full_text, blocks
