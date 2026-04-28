import sys
import os
from loguru import logger

# Add current directory to path so we can import our modules
sys.path.append(os.getcwd())

from fuzzy_match import match_medicine_name, match_manufacturer

def test_fuzzy_matching():
    logger.info("Starting Fuzzy Matching Test...")

    # Test Case 1: Medicine Name with typo
    test_medicines = [
        ("Augmentin 625", "Augmentin 625 Duo Tablet"),
        ("Azithral 500", "Azithral 500 Tablet"),
        ("Allegra 120", "Allegra 120mg Tablet"),
        ("Aciloc 150", "Aciloc 150 Tablet"),
        ("Dolo 650", "Dolo 650 Tablet"),
    ]

    print("\n--- Medicine Name Matching ---")
    for input_name, expected_name in test_medicines:
        result = match_medicine_name([input_name])
        status = "PASS" if result.matched_value == expected_name else "FAIL"
        print(f"Input: '{input_name}'")
        print(f"Match: '{result.matched_value}' (Score: {result.score:.1f}, Confidence: {result.confidence})")
        print(f"Status: {status}\n")

    # Test Case 2: Manufacturer with variations
    test_manufacturers = [
        ("Glaxo SmithKline", "Glaxo SmithKline Pharmaceuticals Ltd"),
        ("Alembic Pharma", "Alembic Pharmaceuticals Ltd"),
        ("Sanofi India", "Sanofi India  Ltd"),
        ("Dr Reddys", "Dr Reddy's Laboratories Ltd"),
    ]

    print("\n--- Manufacturer Matching ---")
    from rapidfuzz import process, fuzz
    from medicine_db import MANUFACTURER_DATABASE
    
    for input_mfr, expected_mfr in test_manufacturers:
        result = match_manufacturer(input_mfr)
        status = "PASS" if result.matched_value == expected_mfr else "FAIL"
        print(f"Input: '{input_mfr}'")
        print(f"Match: '{result.matched_value}' (Score: {result.score:.1f}, Confidence: {result.confidence})")
        
        if status == "FAIL":
            print("Top 5 candidates:")
            top = process.extract(input_mfr, MANUFACTURER_DATABASE, scorer=fuzz.WRatio, limit=5)
            for m, s, i in top:
                print(f"  - {m} (Score: {s:.1f})")
        
        print(f"Status: {status}\n")

if __name__ == "__main__":
    test_fuzzy_matching()
