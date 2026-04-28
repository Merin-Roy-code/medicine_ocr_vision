"""
ocr.py — Google Cloud Vision OCR integration.

Responsibilities:
  1. Send preprocessed image bytes to Google Cloud Vision for full-text extraction.
  2. Return both the full-text string AND a list of TextBlock objects,
     providing spatial bounding boxes for the extraction pipeline.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from google.cloud import vision
from loguru import logger


# ──────────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TextBlock:
    """Represents a single OCR text block."""
    text: str
    confidence: float
    # Bounding box vertices
    vertices: list[tuple[int, int]] = field(default_factory=list)

    @property
    def top_left_y(self) -> int:
        return min(v[1] for v in self.vertices) if self.vertices else 0

    @property
    def top_left_x(self) -> int:
        return min(v[0] for v in self.vertices) if self.vertices else 0


# ──────────────────────────────────────────────────────────────────────────────
# Public OCR function
# ──────────────────────────────────────────────────────────────────────────────

def run_ocr(image_bytes: bytes) -> tuple[str, list[TextBlock]]:
    """
    Send image bytes to Google Cloud Vision and return structured OCR output.

    Args:
        image_bytes: PNG/JPEG bytes of the (preprocessed) image.

    Returns:
        full_text:  The complete text extracted from the image.
        blocks:     List containing individual TextBlocks with bounding boxes.

    Raises:
        RuntimeError: On any Google Cloud Vision API failure.
    """
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)

    logger.info("Calling Google Cloud Vision for OCR …")
    
    try:
        response = client.text_detection(image=image)
    except Exception as exc:
        raise RuntimeError(f"Google Cloud Vision API error: {exc}") from exc

    if response.error.message:
        raise RuntimeError(
            f"Google Cloud Vision API error: {response.error.message}\n"
            f"For more info on error messages, check: "
            f"https://cloud.google.com/apis/design/errors"
        )

    texts = response.text_annotations

    if not texts:
        logger.warning("Google Cloud Vision returned empty text for this image.")
        return "", []

    # The first element in text_annotations is the entire text found in the image.
    full_text = texts[0].description
    blocks = []

    # Iterate from the 1st element onwards to get the individual blocks/words
    for text in texts[1:]:
        vertices = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
        # Cloud Vision text_annotations doesn't provide word-level confidence
        blocks.append(TextBlock(text=text.description, confidence=0.95, vertices=vertices))

    logger.debug(f"Google Cloud Vision OCR output ({len(full_text)} chars):\n{full_text[:300]}")
    logger.info(f"Google Cloud Vision OCR complete — extracted {len(full_text)} characters, {len(blocks)} blocks.")
    
    return full_text, blocks

