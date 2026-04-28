"""
medicine_db.py — DuckDB integration for fast, scalable medicine & manufacturer search.
"""

import os
import re
import duckdb
from loguru import logger

DATASET_PATH = "medicine_dataset.csv"
DB_PATH = "medicine.duckdb"

# Global connection to the DuckDB database (thread-safe for concurrent readers)
_conn = None

def get_connection():
    """Get or create the DuckDB database connection."""
    global _conn
    if _conn is None:
        _conn = duckdb.connect(DB_PATH)
        _init_db(_conn)
    return _conn

def _init_db(conn):
    """Create the table and load data from CSV if it doesn't exist."""
    tables = conn.execute("SHOW TABLES").fetchall()
    tables = [t[0] for t in tables]
    
    if 'medicines' not in tables:
        if not os.path.exists(DATASET_PATH):
            logger.warning(f"Dataset file '{DATASET_PATH}' not found. Cannot initialize DuckDB.")
            # Create an empty table to prevent query errors
            conn.execute('''
                CREATE TABLE medicines (
                    name VARCHAR,
                    manufacturer_name VARCHAR,
                    clean_name VARCHAR
                )
            ''')
            return

        logger.info(f"Initializing DuckDB database from {DATASET_PATH}...")
        
        # 1. Load data from CSV
        # ignore_errors=true helps bypass any malformed rows
        conn.execute(f"""
            CREATE TABLE medicines AS 
            SELECT DISTINCT name, manufacturer_name 
            FROM read_csv_auto('{DATASET_PATH}', ignore_errors=true)
            WHERE name IS NOT NULL AND name != ''
        """)
        
        # 2. Add 'clean_name' column (strip common suffixes like TABLET, SYRUP, dosages)
        logger.info("Generating clean names for base matching...")
        conn.execute("ALTER TABLE medicines ADD COLUMN clean_name VARCHAR")
        conn.execute("UPDATE medicines SET clean_name = upper(name)")
        
        # Strip common form suffixes
        form_pattern = r'\s+(TABLET|SYRUP|CAPSULE|INJECTION|SUSPENSION|TABS|CAPS|INJ|SR|XR|ER|FR|SOFTGEL|GEL|OINTMENT|CREAM|DROPS|LOTION|SOLUTION|POWDER|SACHET|RESPULES|MD|DT|OD|CR|DS|PLUS|FORTE|FORTH|MINI|LITE|NEW|MAX)\b.*'
        conn.execute(f"UPDATE medicines SET clean_name = regexp_replace(clean_name, '{form_pattern}', '', 'g')")
        
        # Strip dosages (e.g., 500MG, 10ML)
        dosage_pattern = r'\s+\d+(MG|MCG|ML|G|IU|%)\b.*'
        conn.execute(f"UPDATE medicines SET clean_name = regexp_replace(clean_name, '{dosage_pattern}', '', 'g')")
        
        conn.execute("UPDATE medicines SET clean_name = trim(clean_name)")
        
        # 3. Create indexes for blazing fast exact searches
        logger.info("Creating indexes...")
        conn.execute("CREATE INDEX idx_name_upper ON medicines(upper(name))")
        conn.execute("CREATE INDEX idx_clean_name ON medicines(clean_name)")
        
        logger.info("DuckDB database initialized successfully.")

# ──────────────────────────────────────────────────────────────────────────────
# Public Search API
# ──────────────────────────────────────────────────────────────────────────────

def get_exact_match(candidate: str) -> tuple[str, str | None] | None:
    """O(1) exact match against the original name, returning (name, manufacturer)."""
    conn = get_connection()
    res = conn.execute("SELECT name, manufacturer_name FROM medicines WHERE upper(name) = ?", (candidate.upper(),)).fetchone()
    return (res[0], res[1]) if res else None

def get_base_matches(candidate: str) -> list[tuple[str, str | None]]:
    """Find all medicines sharing the same clean base name, returning [(name, manufacturer)]."""
    conn = get_connection()
    
    # Strip candidate similarly to how we stripped the database
    clean_cand = candidate.upper()
    clean_cand = re.sub(r'\s+(TABLET|SYRUP|CAPSULE|INJECTION|SUSPENSION|TABS|CAPS|INJ|SR|XR|ER|FR|SOFTGEL|GEL|OINTMENT|CREAM|DROPS|LOTION|SOLUTION|POWDER|SACHET|RESPULES|MD|DT|OD|CR|DS|PLUS|FORTE|FORTH|MINI|LITE|NEW|MAX)\b.*', '', clean_cand)
    clean_cand = re.sub(r'\s+\d+(MG|MCG|ML|G|IU|%)\b.*', '', clean_cand)
    clean_cand = clean_cand.strip()

    res = conn.execute("SELECT name, manufacturer_name FROM medicines WHERE clean_name = ?", (clean_cand,)).fetchall()
    return [(r[0], r[1]) for r in res]

def get_fuzzy_candidates(candidate: str, limit: int = 100) -> list[str]:
    """
    Use DuckDB's Jaro-Winkler similarity to quickly pre-filter candidates
    from the 250k dataset, returning a smaller list for RapidFuzz to evaluate.
    """
    conn = get_connection()
    query = """
        SELECT name, jaro_winkler_similarity(upper(name), ?) as score
        FROM medicines
        WHERE score > 0.6
        ORDER BY score DESC
        LIMIT ?
    """
    res = conn.execute(query, (candidate.upper(), limit)).fetchall()
    return [r[0] for r in res]

def get_fuzzy_manufacturer_candidates(candidate: str, limit: int = 100) -> list[str]:
    """Pre-filter manufacturer names."""
    conn = get_connection()
    query = """
        SELECT DISTINCT manufacturer_name, jaro_winkler_similarity(upper(manufacturer_name), ?) as score
        FROM medicines
        WHERE manufacturer_name IS NOT NULL AND score > 0.5
        ORDER BY score DESC
        LIMIT ?
    """
    res = conn.execute(query, (candidate.upper(), limit)).fetchall()
    return [r[0] for r in res]
