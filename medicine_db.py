"""
medicine_db.py — In-memory medicine & manufacturer database used for fuzzy matching.

In production, replace / extend these lists by loading from a CSV or a database.
The lists are intentionally broad to cover Indian generics, branded drugs, and
common OTC medicines.
"""

import csv
import os
import re
from loguru import logger

# ──────────────────────────────────────────────────────────────────────────────
# Database Loading Logic
# ──────────────────────────────────────────────────────────────────────────────

DATASET_PATH = "medicine_dataset.csv"

def load_database() -> tuple[list[str], list[str], set[str], dict[str, list[str]]]:
    """
    Load medicine names and manufacturer names from the CSV dataset.
    Returns: (medicine_list, manufacturer_list, medicine_set, base_name_map)
    """
    medicines = set()
    manufacturers = set()

    if not os.path.exists(DATASET_PATH):
        logger.warning(f"Dataset file '{DATASET_PATH}' not found. Using empty fallback.")
        return [], [], set(), {}

    logger.info(f"Loading medicine database from '{DATASET_PATH}'...")
    try:
        with open(DATASET_PATH, mode='r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get('name')
                mfr = row.get('manufacturer_name')
                
                if name:
                    name_stripped = name.strip()
                    if name_stripped:
                        medicines.add(name_stripped)
                if mfr:
                    mfr_stripped = mfr.strip()
                    if mfr_stripped:
                        manufacturers.add(mfr_stripped)
        
        logger.info(f"Loaded {len(medicines)} medicines and {len(manufacturers)} manufacturers.")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return [], [], set(), {}

    # Create normalized set for O(1) matching
    medicine_set = {n.upper() for n in medicines}
    
    # Create Base Name Map: Maps base brands to their full canonical names
    # Example: "PARACETAMOL" -> ["Paracetamol 500mg Tablet", "Paracetamol 650mg Tablet"]
    # This works dynamically for ALL 250,000+ medicines in the database.
    base_name_map = {}
    # Common form suffixes to strip to find the "Brand Name"
    suffixes = (
        r'\s+(TABLET|SYRUP|CAPSULE|INJECTION|SUSPENSION|TABS|CAPS|INJ|SR|XR|ER|FR|'
        r'SOFTGEL|GEL|OINTMENT|CREAM|DROPS|LOTION|SOLUTION|POWDER|SACHET|RESPULES|'
        r'MD|DT|OD|CR|DS|PLUS|FORTE|FORTH|MINI|LITE|NEW|MAX)\b.*'
    )
    
    for name in medicines:
        upper = name.upper()
        # Strip common suffixes and dosages (e.g., "500MG")
        base = re.sub(suffixes, '', upper).strip()
        base = re.sub(r'\s+\d+(MG|MCG|ML|G|IU|%)\b.*', '', base).strip()
        
        if base not in base_name_map:
            base_name_map[base] = []
        base_name_map[base].append(name)

    return list(medicines), list(manufacturers), medicine_set, base_name_map

# ──────────────────────────────────────────────────────────────────────────────
# In-memory databases
# ──────────────────────────────────────────────────────────────────────────────

# Initialise databases on module load
MEDICINE_DATABASE, MANUFACTURER_DATABASE, MEDICINE_SET, BASE_NAME_MAP = load_database()
