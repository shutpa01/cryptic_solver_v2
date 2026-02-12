#!/usr/bin/env python3
"""
One-off backfill script to populate the master clues table with all clues
from individual publication tables that don't already exist.

Preserves existing clues (especially those with explanations).
Nothing gets deleted.

Run once, then rely on daily_scraper.py to keep it current.
"""

import sqlite3
from datetime import datetime
from pathlib import Path

# Configuration
DB_PATH = Path(r"C:\Users\shute\PycharmProjects\AI_Solver\data\clues_master.db")

# Publication tables and their column mappings
# Format: (table_name, source_name, date_column)
PUBLICATION_TABLES = [
    ('times_clues', 'times', 'puzzle_date'),
    ('telegraph_clues', 'telegraph', 'puzzle_date'),
    ('guardian_clues', 'guardian', 'puzzle_date'),
    ('independent_clues', 'independent', 'puzzle_date'),
    ('ft_clues', 'ft', 'puzzle_date'),
]


def get_table_columns(cursor, table_name):
    """Get column names for a table."""
    cursor.execute(f"PRAGMA table_info({table_name})")
    return [row[1] for row in cursor.fetchall()]


def backfill_from_table(cursor, table_name: str, source_name: str,
                        date_column: str) -> int:
    """
    Backfill clues from a publication table into master clues table.

    Returns number of rows inserted.
    """
    # Check table exists
    cursor.execute(f"SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
                   (table_name,))
    if cursor.fetchone()[0] == 0:
        print(f"  Table {table_name} does not exist, skipping")
        return 0

    # Get columns from source table
    source_cols = get_table_columns(cursor, table_name)

    # Map source columns to clues table columns
    # We need: puzzle_number, publication_date, clue_number, direction, clue_text, enumeration, answer

    # Build column mapping based on what exists
    col_map = {
        'puzzle_number': 'puzzle_number' if 'puzzle_number' in source_cols else 'NULL',
        'publication_date': date_column if date_column in source_cols else 'NULL',
        'clue_number': 'clue_number' if 'clue_number' in source_cols else 'NULL',
        'direction': 'direction' if 'direction' in source_cols else 'NULL',
        'clue_text': 'clue_text',  # Required
        'enumeration': 'enumeration' if 'enumeration' in source_cols else 'NULL',
        'answer': 'answer',  # Required
    }

    # Count existing in source
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    source_count = cursor.fetchone()[0]
    print(f"  Source table has {source_count:,} rows")

    # Insert clues that don't already exist (match on clue_text + answer)
    insert_sql = f"""
        INSERT INTO clues (
            source, puzzle_number, publication_date, clue_number,
            direction, clue_text, enumeration, answer, original_db, original_id
        )
        SELECT 
            ?,
            {col_map['puzzle_number']},
            {col_map['publication_date']},
            {col_map['clue_number']},
            {col_map['direction']},
            clue_text,
            {col_map['enumeration']},
            answer,
            ?,
            id
        FROM {table_name} src
        WHERE NOT EXISTS (
            SELECT 1 FROM clues c 
            WHERE LOWER(TRIM(c.clue_text)) = LOWER(TRIM(src.clue_text)) 
            AND LOWER(TRIM(c.answer)) = LOWER(TRIM(src.answer))
        )
    """

    cursor.execute(insert_sql, (source_name, table_name))
    inserted = cursor.rowcount

    return inserted


def create_unique_index_if_needed(cursor):
    """Create index for faster duplicate checking if it doesn't exist."""
    # Check if index exists
    cursor.execute("""
        SELECT COUNT(*) FROM sqlite_master 
        WHERE type='index' AND name='idx_clues_text_answer_lower'
    """)

    if cursor.fetchone()[0] == 0:
        print("Creating index for faster lookups...")
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_clues_text_answer_lower 
            ON clues(LOWER(TRIM(clue_text)), LOWER(TRIM(answer)))
        """)
        print("Index created")


def main():
    print(f"Backfill Clues Master - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Database: {DB_PATH}")
    print()

    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get initial count
    cursor.execute("SELECT COUNT(*) FROM clues")
    initial_count = cursor.fetchone()[0]
    print(f"Initial clues count: {initial_count:,}")

    # Count with explanations (preserve these!)
    cursor.execute(
        "SELECT COUNT(*) FROM clues WHERE explanation IS NOT NULL AND explanation != ''")
    explained_count = cursor.fetchone()[0]
    print(f"Clues with explanations: {explained_count:,} (will be preserved)")
    print()

    # Create index for faster duplicate checking
    create_unique_index_if_needed(cursor)

    # Backfill from each publication table
    total_inserted = 0
    for table_name, source_name, date_column in PUBLICATION_TABLES:
        print(f"Processing {table_name}...")
        try:
            inserted = backfill_from_table(cursor, table_name, source_name, date_column)
            total_inserted += inserted
            print(f"  Inserted: {inserted:,} new clues")
        except Exception as e:
            print(f"  ERROR: {e}")
        print()

    # Commit
    conn.commit()

    # Final count
    cursor.execute("SELECT COUNT(*) FROM clues")
    final_count = cursor.fetchone()[0]

    # Verify explanations preserved
    cursor.execute(
        "SELECT COUNT(*) FROM clues WHERE explanation IS NOT NULL AND explanation != ''")
    final_explained = cursor.fetchone()[0]

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Initial clues:     {initial_count:,}")
    print(f"New clues added:   {total_inserted:,}")
    print(f"Final clues:       {final_count:,}")
    print(f"Explanations:      {final_explained:,} (was {explained_count:,})")

    if final_explained < explained_count:
        print("WARNING: Some explanations may have been lost!")
    else:
        print("✓ All explanations preserved")

    conn.close()
    print("\nDone!")


if __name__ == "__main__":
    main()