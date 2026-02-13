#!/usr/bin/env python3
"""One-time reload of the master clues table from publication tables.

Deletes all existing rows in the clues table and reloads from:
  - guardian_clues
  - telegraph_clues
  - times_clues
  - independent_clues

Adds a unique index for future upsert support.
"""

import sqlite3
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

DB_PATH = os.getenv('DB_PATH',
                    r"C:\Users\shute\PycharmProjects\cryptic_solver_V2\data\clues_master.db")

PUBLICATION_TABLES = [
    ('guardian_clues', 'guardian'),
    ('telegraph_clues', 'telegraph'),
    ('times_clues', 'times'),
    ('independent_clues', 'independent'),
]


def main():
    print("=" * 60)
    print("RELOAD MASTER CLUES TABLE")
    print("=" * 60)
    print(f"Database: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Current count
    cursor.execute("SELECT COUNT(*) FROM clues")
    old_count = cursor.fetchone()[0]
    print(f"\nCurrent clues table: {old_count:,} rows")

    # Show current source breakdown
    cursor.execute("SELECT source, COUNT(*) FROM clues GROUP BY source ORDER BY COUNT(*) DESC")
    print("Current sources:")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]:,}")

    # Delete all rows
    print(f"\nDeleting all {old_count:,} rows...")
    cursor.execute("DELETE FROM clues")
    conn.commit()
    print("  Done.")

    # Drop old indexes and create new unique index
    # Use clue_text + answer for dedup since some publications have NULL puzzle_number/direction
    print("\nCreating unique index for dedup...")
    cursor.execute("DROP INDEX IF EXISTS idx_clues_dedup")
    cursor.execute("""
        CREATE UNIQUE INDEX idx_clues_dedup
        ON clues(source, answer, clue_text)
    """)
    conn.commit()
    print("  Done.")

    # Reload from each publication table
    total_inserted = 0

    for table_name, source_name in PUBLICATION_TABLES:
        # Check table exists
        cursor.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,))
        if cursor.fetchone()[0] == 0:
            print(f"\n  {source_name}: table {table_name} not found, skipping")
            continue

        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        src_count = cursor.fetchone()[0]

        print(f"\n  Loading {source_name} from {table_name} ({src_count:,} rows)...")

        cursor.execute(f"""
            INSERT OR IGNORE INTO clues (
                source, puzzle_number, publication_date, clue_number,
                direction, clue_text, enumeration, answer,
                original_db, original_id
            )
            SELECT
                ?,
                puzzle_number,
                puzzle_date,
                clue_number,
                direction,
                clue_text,
                enumeration,
                answer,
                ?,
                id
            FROM {table_name}
        """, (source_name, table_name))

        inserted = cursor.rowcount
        total_inserted += inserted
        print(f"  Inserted: {inserted:,}")

        if inserted < src_count:
            print(f"  Skipped {src_count - inserted:,} duplicates (same source/puzzle/clue/direction)")

    conn.commit()

    # Final count
    cursor.execute("SELECT COUNT(*) FROM clues")
    new_count = cursor.fetchone()[0]

    print(f"\n{'=' * 60}")
    print(f"DONE")
    print(f"  Previous: {old_count:,}")
    print(f"  Inserted: {total_inserted:,}")
    print(f"  New total: {new_count:,}")
    print(f"{'=' * 60}")

    # Verify
    cursor.execute("SELECT source, COUNT(*) FROM clues GROUP BY source ORDER BY COUNT(*) DESC")
    print("\nNew source breakdown:")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]:,}")

    conn.close()


if __name__ == "__main__":
    main()
