"""
config.py — Centralised configuration loaded from environment variables.

Set GEMINI_API_KEY in your .env file before running the server:

  GEMINI_API_KEY=AIzaSy...   (.env file)
"""

import os
from dotenv import load_dotenv

load_dotenv()  # loads .env file if present

# ─── Gemini API ───────────────────────────────────────────────────────────────
# API key for Gemini Vision (required for OCR)
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")

# ─── Preprocessing ─────────────────────────────────────────────────────────────
# Target long-edge size in pixels for upscaling small images
PREPROCESS_TARGET_SIZE: int = int(os.getenv("PREPROCESS_TARGET_SIZE", "1600"))

# CLAHE clip limit for contrast enhancement
CLAHE_CLIP_LIMIT: float = float(os.getenv("CLAHE_CLIP_LIMIT", "2.0"))

# Fast NL Means denoising strengths
DENOISE_H: int = int(os.getenv("DENOISE_H", "6"))
DENOISE_TEMPLATE_WINDOW: int = 7
DENOISE_SEARCH_WINDOW: int = 21

# ─── Fuzzy Matching ─────────────────────────────────────────────────────────────
# Minimum RapidFuzz score (0–100) to accept a fuzzy match
FUZZY_MIN_SCORE: float = float(os.getenv("FUZZY_MIN_SCORE", "70.0"))

# ─── API ───────────────────────────────────────────────────────────────────────
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
