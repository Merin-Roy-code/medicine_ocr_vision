"""
extractor.py — Text cleaning + regex-based field extraction.

Pipeline:
  1. clean_text()   — Normalise case, fix common OCR substitutions, strip noise.
  2. extract_*()    — Field-specific regex functions that are tolerant of OCR
                      errors in keywords (e.g., "BAT CH", "EXD", "MFO").
  3. normalize_date() — Unify all detected date formats to MM/YYYY.
  4. extract_all()  — Orchestrator that ties everything together.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger


# ──────────────────────────────────────────────────────────────────────────────
# Data class returned by the extractor
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RawExtraction:
    """Intermediate result from regex extraction (before fuzzy matching)."""
    medicine_name_candidates: list[str] = field(default_factory=list)
    batch: Optional[str] = None
    expiry: Optional[str] = None
    manufacturer: Optional[str] = None
    composition: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────────
# Step 1 — Text cleaning
# ──────────────────────────────────────────────────────────────────────────────

# Common OCR character-level substitutions to fix before regex matching.
# These are single-character swaps that OCR engines frequently get wrong on
# medicine packaging (metallic/foil backgrounds, small fonts).
_OCR_CHAR_FIXES: list[tuple[str, str]] = [
    # Letter ↔ digit swaps
    # Uncommon symbol noise
    ("|", "I"),
    ("!", "I"),
    (",", "."),
    (";", ":"),
]

# Whole-word / token corrections for mis-scanned keywords
_OCR_KEYWORD_FIXES: dict[str, str] = {
    # Expiry-related
    "EXD": "EXP",
    "EYP": "EXP",
    "EXF": "EXP",
    "EXPI": "EXP",
    "EXPIRY": "EXP",
    "EXP.": "EXP",
    # Batch-related
    "BAT": "BATCH",
    "BATC": "BATCH",
    "BAIC": "BATCH",
    "BAICH": "BATCH",
    "LOIT": "LOT",
    "B.NO": "BATCH",
    "B.NO.": "BATCH",
    # Manufacturer-related
    "MFO": "MFG",
    "MIFG": "MFG",
    "MFD": "MFG",
    "MFD.": "MFG",
    "MANUFACT": "MFG",
    # Misc
    "LTD": "LTD.",
    "LIMILED": "LIMITED",
    "LIMITID": "LIMITED",
}


def clean_text(text: str) -> str:
    """
    Normalise and fix raw OCR text before regex extraction.

    Steps:
      1. Strip leading/trailing whitespace, collapse multiple spaces.
      2. Convert to uppercase (all our regex patterns use uppercase).
      3. Apply character-level OCR fixes.
      4. Apply keyword-level OCR fixes (whole word replacement).
      5. Remove truly junk characters that are never part of valid fields.

    Args:
        text: Raw OCR text from Vision API.

    Returns:
        Cleaned, normalised text string.
    """
    if not text:
        return ""

    # Uppercase
    text = text.upper()

    # Character-level fixes
    for bad, good in _OCR_CHAR_FIXES:
        text = text.replace(bad, good)

    # Keyword-level fixes (whole words only)
    for bad_kw, good_kw in _OCR_KEYWORD_FIXES.items():
        text = re.sub(rf"\b{re.escape(bad_kw)}\b", good_kw, text)

    # Remove truly junk characters but KEEP newlines and spaces
    text = re.sub(r"[^\w\s\.\-\/\:\(\)%+]", " ", text)

    # Collapse multiple spaces but keep newlines
    text = re.sub(r"[ \t]+", " ", text)
    # Strip whitespace from start/end of each line
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())

    logger.debug(f"Cleaned text snippet: {text[:120]} …")
    return text


# ──────────────────────────────────────────────────────────────────────────────
# Step 2 — Field-specific regex extraction helpers
# ──────────────────────────────────────────────────────────────────────────────

def extract_batch(text: str) -> Optional[str]:
    """
    Extract batch / lot number from cleaned text.

    Patterns covered:
      • BATCH NO: NF1234        • BATCH NO . NF1234
      • BATCH: NF1234           • B.NO: NF1234
      • B.NO NF1234             • LOT: NF1234
      • LOT NO NF1234           • B NO NF1234

    Returns the first alphanumeric token after the keyword, or None.
    """
    pattern = re.compile(
        r"""
        (?:
            BATCH\s*(?:NO|NUMBER|NO\.?|\.)?   # BATCH NO / BATCH NUMBER / BATCH.
          | B\.?\s*NO\.?                        # B.NO / B NO
          | LOT\s*(?:NO\.?|NUMBER)?             # LOT / LOT NO / LOT NUMBER
        )
        \s*[:\-\.\s]*\s*                        # separator (colon, dash, dot, space)
        ([A-Z0-9][A-Z0-9\-\/]{2,20})            # batch value: 3–21 alphanumeric chars
        """,
        re.VERBOSE,
    )
    match = pattern.search(text)
    if match:
        value = match.group(1).strip()
        logger.debug(f"Batch extracted: {value}")
        return value
    return None


# Date patterns captured by the expiry regex groups:
# Group 1 — MM/YYYY or MM-YYYY        e.g. 05/2026, 03-2025
# Group 2 — MM/YY or MM-YY            e.g. 05/26
# Group 3 — MonthName YYYY            e.g. MAY 2026
# Group 4 — YYYY-MM-DD ISO            e.g. 2026-05-01

_DATE_PATTERN = re.compile(
    r"""
    (?:
        (0?[1-9]|1[0-2])          # month 01–12
        [\/\-\.]
        (20\d{2}|\d{2})           # year YYYY or YY
    )
    |
    (?:
        (JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)  # month name
        [\s\-\/\.]*
        (20\d{2}|\d{2})           # year YYYY or YY
    )
    |
    (?:
        (20\d{2})                 # ISO: YYYY
        [\/\-\.]
        (0?[1-9]|1[0-2])          # then MM
        (?:[\/\-\.]\d{1,2})?      # optional DD
    )
    """,
    re.VERBOSE,
)

_MONTH_ABBR_MAP: dict[str, str] = {
    "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
    "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
    "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
}


def normalize_date(raw_date: str) -> str:
    """
    Convert any recognised date format to MM/YYYY.

    Examples:
      "05/2026"   → "05/2026"
      "05/26"     → "05/2026"
      "MAY 2026"  → "05/2026"
      "2026-05"   → "05/2026"
    """
    raw_date = raw_date.strip().upper()

    m = _DATE_PATTERN.search(raw_date)
    if not m:
        return raw_date  # cannot normalise — return as-is

    g = m.groups()

    if g[0] and g[1]:
        # Group 1+2: MM / YYYY or MM / YY
        month = g[0].zfill(2)
        year = g[1] if len(g[1]) == 4 else f"20{g[1]}"
        return f"{month}/{year}"

    if g[2] and g[3]:
        # Group 3+4: MonthName YYYY
        month = _MONTH_ABBR_MAP.get(g[2], "??")
        return f"{month}/{g[3]}"

    if g[4] and g[5]:
        # Group 5+6: YYYY-MM (ISO-like)
        month = g[5].zfill(2)
        return f"{month}/{g[4]}"

    return raw_date


def extract_expiry(text: str) -> Optional[str]:
    """
    Extract expiry date from cleaned text.

    Patterns covered:
      • EXP: 05/2026          • EXP. DATE: 05/2026
      • EXPIRY: MAY 2026      • USE BEFORE: 05/26
      • USE BY: 05/2026       • BEST BEFORE: MAY 2026
      • MFG: 01/2024 EXP: 05/2026  (handles adjacent MFG/EXP labels)

    Returns: Normalised date string (MM/YYYY) or None.
    """
    pattern = re.compile(
        r"""
        (?:
            EXP(?:IRY|IRATION)?       # EXP / EXPIRY / EXPIRATION
          | USE\s*(?:BEFORE|BY)       # USE BEFORE / USE BY
          | BEST\s*BEFORE             # BEST BEFORE
        )
        \.?\s*(?:DATE\.?)?\s*[:\-]?\s*  # optional "DATE", optional separator
        (                               # capture date value
            (?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[\s\-\/\.]*(?:20\d{2}|\d{2})
          | 0?[1-9][\/\-\.](?:20\d{2}|\d{2})
          | 1[0-2][\/\-\.](?:20\d{2}|\d{2})
          | 20\d{2}[\/\-\.](?:0?[1-9]|1[0-2])
        )
        """,
        re.VERBOSE,
    )
    match = pattern.search(text)
    if match:
        raw = match.group(1).strip()
        normalised = normalize_date(raw)
        logger.debug(f"Expiry extracted: {raw} → {normalised}")
        return normalised
    return None


def extract_composition(text: str) -> Optional[str]:
    """
    Extract chemical composition / active ingredients.
    
    Looks for patterns like:
      - Each film-coated tablet contains: Cetirizine ...
      - Composition: ...
      - Rx [Ingredient Name]
    """
    # 1. Look for "contains:" or "Composition:"
    pattern_contains = re.compile(
        r"(?:CONTAINS|COMPOSITION)\s*[:\-]?\s*([A-Z][A-Z0-9\s\.\(\)%]{5,100})",
        re.IGNORECASE | re.MULTILINE
    )
    
    # 2. Look for "Rx [Name]" (often found on prescription drug headings)
    pattern_rx = re.compile(
        r"RX\s+([A-Z][A-Z\s]{5,40}(?:HYDROCHLORIDE|SULPHATE|MALEATE|PHOSPHATE|IP|BP|USP)?)",
        re.IGNORECASE
    )

    match = pattern_contains.search(text) or pattern_rx.search(text)
    if match:
        val = match.group(1).strip()
        # Clean up noise like dots used as fillers (e.g., "....... 10 mg")
        val = re.sub(r"\.{2,}", " ", val)
        val = re.sub(r"\s+", " ", val).strip()
        
        # Cut off the string if it bleeds into the next section
        val = re.split(r'\b(?:COLOUR|COLOR|DOSAGE|STORE|KEEP|WARNING|CAUTION|MFD)\b', val, maxsplit=1, flags=re.IGNORECASE)[0].strip()
        
        # Cap length
        return val[:80]
    return None


def extract_manufacturer(text: str) -> Optional[str]:
    """
    Extract manufacturer / marketer name from cleaned text.

    Patterns covered:
      • MANUFACTURED BY: Cipla Ltd.
      • MFG BY: ...
      • MFG: ...
      • MARKETED BY: ...
      • MARKETED AND DISTRIBUTED BY: ...
      • DISTRIBUTED BY: ...

    Returns the text snippet (up to 60 chars) after the keyword, or None.
    """
    pattern = re.compile(
        r"""
        (?:
            MFG(?:\s*BY)?               # MFG / MFG BY
          | MANUFACTURED\s*BY          # MANUFACTURED BY
          | MARKETED\s*(?:AND\s*(?:DISTRIBUTED\s*BY|BY))?  # MARKETED BY / AND ...
          | DISTRIBUTED\s*BY           # DISTRIBUTED BY
          | MKTD\s*BY                  # MKTD BY (abbreviation)
        )
        \.?\s*[:\-]?\s*                # separator
        (?!JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC|\d{2}[\/\.]) # Lookahead: Don't start with a date
        ([A-Z][A-Z0-9\s\.\,\-\(\)]{4,60}?)  # manufacturer name (4–60 chars)
        (?=\s{2,}|\n|BATCH|EXP|LOT|$)       # stop at double space, newline, or keywords
        """,
        re.VERBOSE,
    )
    match = pattern.search(text)
    if match:
        value = match.group(1).strip().rstrip(".,")
        # Title-case the extracted manufacturer name for readability
        value = " ".join(w.capitalize() for w in value.split())
        logger.debug(f"Manufacturer extracted: {value}")
        return value
    return None


def extract_medicine_name_candidates(text: str) -> list[str]:
    """
    Heuristically extract medicine name candidates from the OCR text.

    Medicine names appear prominently on packaging — usually the largest text,
    near the top, with a dosage suffix (e.g. "Paracetamol 500 mg").

    Strategy:
      1. Split cleaned text into tokens by newlines / double-spaces.
      2. Find tokens that look like medicine names: start with a letter,
         3–50 chars, may contain digits (dosage).
      3. Exclude known label keywords (BATCH, EXP, MFG, etc.).
      4. Return top candidates for fuzzy matching.
    """
    _EXCLUDE_KEYWORDS = {
        "BATCH", "EXP", "LOT", "MFG", "USE", "BEFORE", "BY",
        "MANUFACTURED", "MARKETED", "DISTRIBUTED", "DATE", "NO",
        "TABLETS", "CAPSULES", "SYRUP", "INJECTION", "STORE",
        "KEEP", "OUT", "REACH", "CHILDREN", "PRESCRIPTION", "MEDICINE",
        "STRIP", "ONLY", "SCHEDULE", "DRUG", "INDIA", "LIMITED", "LTD",
        "M.L.", "ML", "LIC", "LICENCE", "LICENSE", "MFD", "MRB", "MRP", "INCL", "TAXES",
    }

    # Split on newlines or multiple spaces to get logical chunks
    chunks = re.split(r"\n|  +", text)

    candidates: list[str] = []
    for chunk in chunks:
        chunk = chunk.strip()
        
        # 1. Basic length and character filter
        if not re.match(r"^[A-Z][A-Z0-9\s\-\+\.\/]{2,49}$", chunk):
            continue
            
        # 2. Exclude license-like patterns (e.g., M.L. M/485/08, Lic No. 123)
        if re.search(r"\b(LIC|ML|M\.L\.|MFD|EXP|BATCH)\b", chunk):
            # If the chunk contains these keywords but isn't ONLY these keywords,
            # we check if it looks more like a label than a name.
            if any(kw in chunk for kw in ["M.L.", "LIC", "BATCH", "EXP", "MFG"]):
                continue

        # 3. Exclude if all words are keywords
        words = set(re.findall(r"\b\w+\b", chunk))
        if not words or words.issubset(_EXCLUDE_KEYWORDS):
            continue
            
        # 4. Skip chunks that look like addresses or long numbers
        if re.search(r"\d{6}", chunk) or re.search(r"^[0-9\/\-]+$", chunk):
            continue

        candidates.append(chunk)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_candidates: list[str] = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique_candidates.append(c)

    logger.debug(f"Medicine name candidates: {unique_candidates[:5]}")
    return unique_candidates[:10]  # return top 10 for fuzzy matching


# ──────────────────────────────────────────────────────────────────────────────
# Step 3 — Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def extract_all(full_text: str, block_texts: list[str]) -> RawExtraction:
    """
    Run the full extraction pipeline on OCR output.

    Args:
        full_text:   The complete OCR string (all blocks joined).
        block_texts: List of individual block text strings (for per-block hints).

    Returns:
        RawExtraction dataclass with all extracted fields.
    """
    # Clean the full text for global extraction
    cleaned = clean_text(full_text)

    # For manufacturer and batch we also try to search block-by-block to catch
    # cases where Vision split them across blocks in an unexpected way.
    combined_blocks = " \n ".join(clean_text(b) for b in block_texts)

    result = RawExtraction()
    result.batch = extract_batch(cleaned) or extract_batch(combined_blocks)
    result.expiry = extract_expiry(cleaned) or extract_expiry(combined_blocks)
    result.manufacturer = extract_manufacturer(cleaned) or extract_manufacturer(combined_blocks)
    result.composition = extract_composition(cleaned) or extract_composition(combined_blocks)
    result.medicine_name_candidates = extract_medicine_name_candidates(cleaned)

    logger.info(
        f"Regex extraction — batch={result.batch}, expiry={result.expiry}, "
        f"manufacturer={result.manufacturer}, "
        f"name_candidates={len(result.medicine_name_candidates)}"
    )
    return result
