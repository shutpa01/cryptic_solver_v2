#!/usr/bin/env python3
"""Times Crossword Backfill Download

Reads times_api_mapping.json and downloads puzzle data to times_clues table.
"""

import json
import sqlite3
import requests
import time
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent
HARVEST_FILE = SCRIPT_DIR / "times_api_mapping.json"
DB_PATH = Path(r"C:\Users\shute\PycharmProjects\AI_Solver\data\clues_master.db")


def load_harvest():
    """Load harvested API URLs."""
    if not HARVEST_FILE.exists():
        print(f"ERROR: {HARVEST_FILE} not found!")
        return {}
    with open(HARVEST_FILE, 'r') as f:
        return json.load(f)


def init_db(conn, drop_existing=False):
    """Create times_clues table if it doesn't exist."""
    cursor = conn.cursor()

    if drop_existing:
        print("  Dropping existing times_clues table...")
        cursor.execute("DROP TABLE IF EXISTS times_clues")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS times_clues (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            puzzle_type TEXT,
            puzzle_number INTEGER,
            puzzle_date TEXT,
            clue_number TEXT,
            clue_text TEXT,
            answer TEXT,
            direction TEXT,
            enumeration TEXT,
            setter TEXT,
            api_url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(puzzle_type, puzzle_number, clue_number, direction)
        )
    """)
    conn.commit()


def get_existing_puzzles(conn):
    """Get set of (puzzle_type, puzzle_number) already in database."""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT puzzle_type, puzzle_number FROM times_clues
    """)
    return set((row[0], row[1]) for row in cursor.fetchall())


def fetch_puzzle_data(url):
    """Fetch puzzle data from API URL."""
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            return resp.json()
        else:
            print(f"    HTTP {resp.status_code}")
            return None
    except Exception as e:
        print(f"    Error: {e}")
        return None


def parse_and_save_puzzle(conn, puzzle_type, puzzle_number, data, api_url):
    """Parse puzzle data and save clues to database."""
    cursor = conn.cursor()

    # Extract puzzle metadata
    puzzle_data = data.get('data', {})
    copy_data = puzzle_data.get('copy', {})

    puzzle_date = copy_data.get('date-publish', '')
    setter = copy_data.get('setter', '') or copy_data.get('byline', '')

    # Parse clues - they're in copy.clues
    clues_groups = copy_data.get('clues', [])

    clues_saved = 0

    for clue_group in clues_groups:
        direction = clue_group.get('title', '').lower()  # 'Across' or 'Down'
        clues = clue_group.get('clues', [])

        for clue in clues:
            clue_number = clue.get('number', '')
            clue_text = clue.get('clue', '')
            answer = clue.get('answer', '').upper()
            enumeration = clue.get('format', '')  # e.g. "5,8" or "4"

            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO times_clues 
                    (puzzle_type, puzzle_number, puzzle_date, clue_number, clue_text, 
                     answer, direction, enumeration, setter, api_url)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    puzzle_type, puzzle_number, puzzle_date, str(clue_number), clue_text,
                    answer, direction, enumeration, setter, api_url
                ))
                if cursor.rowcount > 0:
                    clues_saved += 1
            except Exception as e:
                print(f"    DB error: {e}")

    conn.commit()
    return clues_saved


def main():
    print("=" * 60)
    print("TIMES CROSSWORD BACKFILL DOWNLOAD")
    print(f"Date: {datetime.now().strftime('%A, %d %B %Y %H:%M')}")
    print("=" * 60)

    # Load harvest
    harvest = load_harvest()
    if not harvest:
        return

    total_urls = sum(len(urls) for urls in harvest.values())
    print(f"\nLoaded {total_urls} API URLs from harvest file")

    # Connect to database
    print(f"\nConnecting to database: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)

    # Check if we should recreate table (first run or schema change)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='times_clues'")
    table_exists = cursor.fetchone() is not None

    if table_exists:
        # Check if it has the right columns
        cursor.execute("PRAGMA table_info(times_clues)")
        columns = [row[1] for row in cursor.fetchall()]
        if 'api_url' not in columns or 'enumeration' not in columns:
            print("  Table has wrong schema, recreating...")
            init_db(conn, drop_existing=True)
        else:
            init_db(conn, drop_existing=False)
    else:
        init_db(conn, drop_existing=False)

    # Get existing puzzles
    existing = get_existing_puzzles(conn)
    print(f"Already have {len(existing)} puzzles in database")

    # Download and save
    print("\n" + "=" * 60)
    print("DOWNLOADING PUZZLES")
    print("=" * 60)

    total_downloaded = 0
    total_clues = 0
    total_skipped = 0
    total_errors = 0

    for puzzle_type, urls in harvest.items():
        print(f"\n--- {puzzle_type.upper()} ({len(urls)} puzzles) ---")

        for puzzle_number, api_url in urls.items():
            # Skip if already in database
            if (puzzle_type, int(puzzle_number)) in existing:
                total_skipped += 1
                continue

            print(f"  #{puzzle_number}...", end=" ", flush=True)

            # Fetch data
            data = fetch_puzzle_data(api_url)

            if data:
                # Parse and save
                clues_saved = parse_and_save_puzzle(
                    conn, puzzle_type, int(puzzle_number), data, api_url
                )
                print(f"✓ {clues_saved} clues")
                total_downloaded += 1
                total_clues += clues_saved
            else:
                print("✗ Failed")
                total_errors += 1

            # Be nice to the server
            time.sleep(0.5)

    conn.close()

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"Downloaded: {total_downloaded} puzzles ({total_clues} clues)")
    print(f"Skipped (already in DB): {total_skipped}")
    print(f"Errors: {total_errors}")


if __name__ == "__main__":
    main()