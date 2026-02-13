#!/usr/bin/env python3
"""Daily Crossword Scraper Orchestrator

Calls each scraper script, then syncs new clues to the master clues table.
Each script handles its own logic for what puzzles to fetch.
"""

import subprocess
import sys
import sqlite3
from datetime import datetime
from pathlib import Path

BASE_PATH = Path(r"C:\Users\shute\PycharmProjects\cryptic_solver_V2\scraper")
PYTHON = r"C:\Users\shute\PycharmProjects\cryptic_solver_V2\.venv\Scripts\python.exe"

# Database paths
CLUES_MASTER_DB = Path(r"C:\Users\shute\PycharmProjects\cryptic_solver_V2\data\clues_master.db")

SCRAPERS = [
    # Telegraph: login, harvest today's API IDs, fetch puzzles
    (BASE_PATH / "telegraph" / "telegraph_daily.py", []),
    # Other newspapers
    (BASE_PATH / "guardian" / "guardian_all.py", []),
    (BASE_PATH / "times" / "times_all.py", []),
    (BASE_PATH / "independent" / "independent_all.py", []),
]

# Publication tables to sync from (table_name, source_name, date_column)
PUBLICATION_TABLES = [
    ('times_clues', 'times', 'puzzle_date'),
    ('telegraph_clues', 'telegraph', 'puzzle_date'),
    ('guardian_clues', 'guardian', 'puzzle_date'),
    ('independent_clues', 'independent', 'puzzle_date'),
]


def run_scraper(script_path: Path, args: list = None):
    """Run a scraper and print its output."""
    print(f"\n{'=' * 60}")
    print(f"RUNNING: {script_path.name} {' '.join(args or [])}")
    print(f"{'=' * 60}")

    cmd = [PYTHON, str(script_path)]
    if args:
        cmd.extend(args)

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=script_path.parent
    )

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    return result.returncode == 0


def sync_to_master_clues():
    """
    Sync all publication tables to the master clues table.
    Uses INSERT OR IGNORE with unique index on (source, puzzle_number, clue_number, direction).
    """
    print(f"\n{'=' * 60}")
    print("SYNCING TO MASTER CLUES TABLE")
    print(f"{'=' * 60}")

    if not CLUES_MASTER_DB.exists():
        print(f"WARNING: Master DB not found at {CLUES_MASTER_DB}")
        return

    conn = sqlite3.connect(CLUES_MASTER_DB)
    cursor = conn.cursor()

    # Ensure unique index exists (source + answer + clue_text handles all publications)
    cursor.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_clues_dedup
        ON clues(source, answer, clue_text)
    """)

    # Get initial count
    cursor.execute("SELECT COUNT(*) FROM clues")
    initial_count = cursor.fetchone()[0]

    total_inserted = 0

    for table_name, source_name, date_column in PUBLICATION_TABLES:
        # Check table exists
        cursor.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        )
        if cursor.fetchone()[0] == 0:
            continue

        insert_sql = f"""
            INSERT OR IGNORE INTO clues (
                source, puzzle_number, publication_date, clue_number,
                direction, clue_text, enumeration, answer, original_db, original_id
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
        """

        try:
            cursor.execute(insert_sql, (source_name, table_name))
            inserted = cursor.rowcount
            if inserted > 0:
                print(f"  {source_name}: +{inserted} new clues")
                total_inserted += inserted
            else:
                print(f"  {source_name}: up to date")
        except Exception as e:
            print(f"  {source_name}: ERROR - {e}")

    conn.commit()

    # Final count
    cursor.execute("SELECT COUNT(*) FROM clues")
    final_count = cursor.fetchone()[0]

    print(f"\nMaster clues: {initial_count:,} -> {final_count:,} (+{total_inserted})")

    conn.close()


def get_sync_stats():
    """Print statistics about sync coverage."""
    if not CLUES_MASTER_DB.exists():
        return

    conn = sqlite3.connect(CLUES_MASTER_DB)
    cursor = conn.cursor()

    print(f"\n{'=' * 60}")
    print("SYNC STATISTICS")
    print(f"{'=' * 60}")

    # Total clues
    cursor.execute("SELECT COUNT(*) FROM clues")
    total = cursor.fetchone()[0]
    print(f"Total clues in master: {total:,}")

    # By source
    cursor.execute("""
        SELECT source, COUNT(*) as cnt 
        FROM clues 
        GROUP BY source 
        ORDER BY cnt DESC
    """)
    print("\nBy source:")
    for row in cursor.fetchall():
        print(f"  {row[0]}: {row[1]:,}")

    # With explanations
    cursor.execute("""
        SELECT COUNT(*) FROM clues 
        WHERE explanation IS NOT NULL AND explanation != ''
    """)
    explained = cursor.fetchone()[0]
    print(f"\nWith human explanations: {explained:,}")

    # Recent clues (last 7 days)
    cursor.execute("""
        SELECT COUNT(*) FROM clues 
        WHERE publication_date >= date('now', '-7 days')
    """)
    recent = cursor.fetchone()[0]
    print(f"Added in last 7 days: {recent:,}")

    conn.close()


def main():
    print(f"Daily Scraper - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Run all scrapers
    for item in SCRAPERS:
        if isinstance(item, tuple):
            script, args = item
        else:
            script, args = item, []

        if script.exists():
            run_scraper(script, args)
        else:
            print(f"Script not found: {script}")

    # Sync to master clues table
    sync_to_master_clues()

    # Print stats
    get_sync_stats()

    print(f"\n{'=' * 60}")
    print("DONE")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()