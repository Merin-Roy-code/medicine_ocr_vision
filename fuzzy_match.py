"""
fuzzy_match.py — Simplified fuzzy matching for medicine names & manufacturers.
REVERTED: Replaced complex multi-stage matching with a stable, faster version.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional

from rapidfuzz import fuzz, process
from loguru import logger

import config
from medicine_db import (
    get_exact_match,
    get_base_matches,
    get_fuzzy_candidates,
    get_fuzzy_manufacturer_candidates
)


@dataclass
class FuzzyMatchResult:
    """Result of a fuzzy match operation."""
    matched_value: Optional[str]
    confidence: float
    raw_input: str
    score: float
    method: str = "none"
    db_manufacturer: Optional[str] = None


def match_medicine_name(
    candidates: list[str],
    min_score: float = config.FUZZY_MIN_SCORE,
) -> FuzzyMatchResult:
    """
    Find the best match for medicine candidates.
    1. Check for exact match in DuckDB (O(1)).
    2. Check for base name match in DuckDB (O(1)).
    3. If no match found, rely on OCR candidate as-is (Bypassed Fuzzy Matching).
    """
    if not candidates:
        return FuzzyMatchResult(None, 0.0, "", 0.0, "none")

    # Step 1 & 2: Exact / Base Name Match (Instant SQL)
    for candidate in candidates:
        upper = candidate.strip().upper()
        if not upper:
            continue
            
        # Case A: Perfect exact match
        exact_res = get_exact_match(upper)
        if exact_res:
            name, mfr = exact_res
            logger.info(f"Perfect exact match found for '{upper}'")
            return FuzzyMatchResult(name.title(), 1.0, candidate, 100.0, "exact", db_manufacturer=mfr)
            
        # Case B: Base name match (e.g., "ALERID" -> "Alerid Tablet")
        matches = get_base_matches(upper)
        if matches:
            # Detect form - for strips, we primarily look for Tablet or Capsule
            detected_form = None
            for form in ["TABLET", "CAPSULE", "TABS", "CAPS"]:
                if form in upper or any(form in c.upper() for c in candidates):
                    detected_form = "TABLET" if "TAB" in form else "CAPSULE"
                    break
            
            # Since this is strictly for medicine strips, we prioritize Tablet/Capsule
            priority_form = detected_form or "TABLET"
            priority_matches = [m for m in matches if priority_form in m[0].upper()]
            
            # Select the best canonical name tuple
            canonical_tuple = sorted(priority_matches or matches, key=lambda x: len(x[0]))[0]
            canonical_name, canonical_mfr = canonical_tuple
            
            logger.info(f"Strip match found: '{upper}' -> '{canonical_name}' (form={priority_form})")
            return FuzzyMatchResult(canonical_name.title(), 0.95, candidate, 95.0, "strip_base_map", db_manufacturer=canonical_mfr)

    # If OCR reads correctly but we couldn't map to database, we bypass fuzzy matching
    # and simply use the first raw candidate as the medicine name.
    best_raw = candidates[0].strip()
    return FuzzyMatchResult(best_raw.title(), 0.90, best_raw, 90.0, "ocr_raw")


def match_manufacturer(
    raw_manufacturer: Optional[str],
    min_score: float = config.FUZZY_MIN_SCORE - 10,
) -> FuzzyMatchResult:
    """
    Avoid fuzzy matching as OCR reads correctly. Return the raw string directly.
    """
    if not raw_manufacturer:
        return FuzzyMatchResult(None, 0.0, "", 0.0)

    raw_clean = raw_manufacturer.strip()
    return FuzzyMatchResult(raw_clean.title(), 0.90, raw_clean, 90.0, "ocr_raw")


async def run_fuzzy_matching_async(
    medicine_candidates: list[str],
    raw_manufacturer: Optional[str],
) -> tuple[FuzzyMatchResult, FuzzyMatchResult]:
    """Execute fuzzy matching for both fields in parallel."""
    # Use to_thread for CPU-bound RapidFuzz tasks to avoid blocking the event loop
    med_task = asyncio.to_thread(match_medicine_name, medicine_candidates)
    mfr_task = asyncio.to_thread(match_manufacturer, raw_manufacturer)
    
    return await asyncio.gather(med_task, mfr_task)


def run_fuzzy_matching(
    medicine_candidates: list[str],
    raw_manufacturer: Optional[str],
) -> tuple[FuzzyMatchResult, FuzzyMatchResult]:
    """Sync wrapper for run_fuzzy_matching_async."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(run_fuzzy_matching_async(medicine_candidates, raw_manufacturer))