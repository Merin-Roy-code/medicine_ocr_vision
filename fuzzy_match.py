"""
fuzzy_match.py — Simplified fuzzy matching for medicine names & manufacturers.
REVERTED: Replaced complex multi-stage matching with a stable, faster version.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rapidfuzz import fuzz, process
from loguru import logger

import config
from medicine_db import MEDICINE_DATABASE, MANUFACTURER_DATABASE, MEDICINE_SET, BASE_NAME_MAP


@dataclass
class FuzzyMatchResult:
    """Result of a fuzzy match operation."""
    matched_value: Optional[str]
    confidence: float
    raw_input: str
    score: float
    method: str = "none"


def match_medicine_name(
    candidates: list[str],
    database: list[str] = MEDICINE_DATABASE,
    min_score: float = config.FUZZY_MIN_SCORE,
) -> FuzzyMatchResult:
    """
    Find the best match for medicine candidates.
    1. Check for exact match in the normalized MEDICINE_SET (O(1)).
    2. Check for exact match in BASE_NAME_MAP (O(1)).
    3. Fall back to fuzzy matching (WRatio) on top candidate.
    """
    if not candidates or not database:
        return FuzzyMatchResult(None, 0.0, "", 0.0, "none")

    best_match_value: Optional[str] = None
    best_score: float = 0.0
    best_raw: str = ""
    best_method: str = "none"

    # Step 1: Exact / Base Name Match (Instant O(1))
    for candidate in candidates:
        upper = candidate.strip().upper()
        if not upper:
            continue
            
        # Case A: Perfect exact match
        if upper in MEDICINE_SET:
            logger.info(f"Perfect exact match found for '{upper}'")
            return FuzzyMatchResult(candidate.title(), 1.0, candidate, 100.0, "exact")
            
        # Case B: Base name match (e.g., "ALERID" -> "Alerid Tablet")
        if upper in BASE_NAME_MAP:
            matches = BASE_NAME_MAP[upper]
            
            # Detect form - for strips, we primarily look for Tablet or Capsule
            detected_form = None
            for form in ["TABLET", "CAPSULE", "TABS", "CAPS"]:
                if form in upper or any(form in c.upper() for c in candidates):
                    detected_form = "TABLET" if "TAB" in form else "CAPSULE"
                    break
            
            # Since this is strictly for medicine strips, we prioritize Tablet/Capsule
            priority_form = detected_form or "TABLET"
            priority_matches = [m for m in matches if priority_form in m.upper()]
            
            # Select the best canonical name
            canonical = sorted(priority_matches or matches, key=len)[0]
            logger.info(f"Strip match found: '{upper}' -> '{canonical}' (form={priority_form})")
            return FuzzyMatchResult(canonical, 0.95, candidate, 95.0, "strip_base_map")

    # Step 2: Fuzzy Match (Fallback - Slow)
    # To keep it fast, we only fuzzy match the first 3 candidates
    for candidate in candidates[:3]:
        candidate = candidate.strip()
        if not candidate or len(candidate) < 3:
            continue

        result = process.extractOne(
            candidate,
            database,
            scorer=fuzz.WRatio,
            score_cutoff=min_score,
        )

        if result:
            matched, score, _ = result
            if score > best_score:
                best_score = score
                best_match_value = matched
                best_raw = candidate
                best_method = "fuzzy"
            
            if score >= 95:
                break

    if best_score < min_score:
        return FuzzyMatchResult(None, round(best_score / 100, 4), best_raw, best_score, best_method)

    return FuzzyMatchResult(best_match_value, round(best_score / 100, 4), best_raw, best_score, best_method)


def match_manufacturer(
    raw_manufacturer: Optional[str],
    database: list[str] = MANUFACTURER_DATABASE,
    min_score: float = config.FUZZY_MIN_SCORE - 10,
) -> FuzzyMatchResult:
    """
    Fuzzy-match manufacturer name using token_set_ratio.
    """
    if not raw_manufacturer:
        return FuzzyMatchResult(None, 0.0, "", 0.0)

    result = process.extractOne(
        raw_manufacturer.strip(),
        database,
        scorer=fuzz.token_set_ratio,
        score_cutoff=min_score - 1,
    )

    if result is None:
        # If no match, return the raw value as fallback
        return FuzzyMatchResult(raw_manufacturer, 0.5, raw_manufacturer, 0.0)

    matched, score, _ = result
    return FuzzyMatchResult(matched, round(score / 100, 4), raw_manufacturer, float(score))


def run_fuzzy_matching(
    medicine_candidates: list[str],
    raw_manufacturer: Optional[str],
) -> tuple[FuzzyMatchResult, FuzzyMatchResult]:
    """Execute fuzzy matching for both fields."""
    medicine_result = match_medicine_name(medicine_candidates)
    manufacturer_result = match_manufacturer(raw_manufacturer)
    return medicine_result, manufacturer_result