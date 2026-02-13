#!/usr/bin/env python3
"""Guardian Gap-Fill Script

Finds missing puzzle numbers in the guardian_clues table and fetches them.
Tries each gap as cryptic first, then prize (since they share a number sequence).
"""

import sys
import os
import time
import sqlite3
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Import functions from guardian_all (same directory)
sys.path.insert(0, str(Path(__file__).parent))
from guardian_all import parse_puzzle, save_to_database, DB_PATH

GUARDIAN_API = "https://www.theguardian.com/crosswords"
TYPES_TO_TRY = ['cryptic', 'prize']
REQUEST_DELAY = 0.5  # seconds between API calls


def find_gaps():
    """Find missing puzzle numbers in the cryptic sequence."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get all distinct puzzle numbers across cryptic and prize types
    cursor.execute("""
        SELECT DISTINCT CAST(puzzle_number AS INTEGER) as pn
        FROM guardian_clues
        WHERE puzzle_type IN ('cryptic', 'prize')
        ORDER BY pn
    """)
    existing = set(r[0] for r in cursor.fetchall())
    conn.close()

    if not existing:
        print("No existing puzzles found.")
        return []

    min_num = min(existing)
    max_num = max(existing)

    gaps = [n for n in range(min_num, max_num + 1) if n not in existing]
    return gaps


def fetch_and_save(puzzle_number):
    """Try to fetch a puzzle number as cryptic, then prize."""
    for puzzle_type in TYPES_TO_TRY:
        url = f"{GUARDIAN_API}/{puzzle_type}/{puzzle_number}.json"

        try:
            response = requests.get(url, timeout=15)
        except Exception as e:
            print(f"  {puzzle_number}: error - {e}")
            return 'failed'

        if response.status_code == 200:
            data = response.json()
            puzzle = parse_puzzle(data, puzzle_type)
            clue_count = save_to_database(puzzle, puzzle_type)
            print(f"  {puzzle_number}: {puzzle_type} - {clue_count} clues")
            return 'fetched'

    print(f"  {puzzle_number}: not found")
    return 'not_found'


def main():
    print("=" * 60)
    print("GUARDIAN GAP-FILL")
    print("=" * 60)
    print(f"Database: {DB_PATH}")

    gaps = find_gaps()
    print(f"Missing puzzle numbers: {len(gaps)}")

    if not gaps:
        print("No gaps to fill!")
        return

    print(f"Range: {gaps[0]} to {gaps[-1]}")
    print()

    stats = {'fetched': 0, 'not_found': 0, 'failed': 0}

    for i, num in enumerate(gaps):
        if i > 0 and i % 100 == 0:
            print(f"\n--- Progress: {i}/{len(gaps)} "
                  f"(fetched={stats['fetched']}, "
                  f"not_found={stats['not_found']}) ---\n")

        result = fetch_and_save(num)
        stats[result] += 1

        if result == 'fetched':
            time.sleep(REQUEST_DELAY)

    print(f"\n{'=' * 60}")
    print(f"DONE")
    print(f"  Fetched:   {stats['fetched']}")
    print(f"  Not found: {stats['not_found']}")
    print(f"  Failed:    {stats['failed']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
