#!/usr/bin/env python3
"""Independent Gap-Fill Script

Finds missing dates in the independent_clues table and fetches them.
Independent publishes Mon-Sat (no Sunday).
"""

import sys
import time
import sqlite3
from datetime import date, timedelta
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Import functions from independent_all (same directory)
sys.path.insert(0, str(Path(__file__).parent))
from independent_all import scrape_puzzle, DB_PATH

REQUEST_DELAY = 1  # seconds between requests


def find_missing_dates():
    """Find missing Mon-Sat dates from earliest entry to today."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get all existing ISO dates
    cursor.execute("""
        SELECT DISTINCT puzzle_date
        FROM independent_clues
        WHERE puzzle_date LIKE '____-__-__'
    """)
    existing = set(r[0] for r in cursor.fetchall())
    conn.close()

    if not existing:
        print("No existing puzzles found.")
        return []

    from datetime import datetime
    sorted_dates = sorted(existing)
    start = datetime.strptime(sorted_dates[0], '%Y-%m-%d').date()
    end = date.today()

    missing = []
    current = start
    while current <= end:
        ds = current.strftime('%Y-%m-%d')
        # Independent is Mon-Sat (weekday 0-5), no Sunday (6)
        if current.weekday() < 6 and ds not in existing:
            missing.append(current)
        current += timedelta(days=1)

    return missing


def main():
    print("=" * 60)
    print("INDEPENDENT GAP-FILL")
    print("=" * 60)
    print(f"Database: {DB_PATH}")

    missing = find_missing_dates()
    print(f"Missing dates (Mon-Sat): {len(missing)}")

    if not missing:
        print("No gaps to fill!")
        return

    print(f"Range: {missing[0]} to {missing[-1]}")
    print()

    total_saved = 0
    total_not_found = 0

    for i, puzzle_date in enumerate(missing):
        if i > 0 and i % 50 == 0:
            print(f"\n--- Progress: {i}/{len(missing)} "
                  f"(saved={total_saved}, not_found={total_not_found}) ---\n")

        count = scrape_puzzle(puzzle_date)
        if count > 0:
            total_saved += count
            time.sleep(REQUEST_DELAY)
        else:
            total_not_found += 1

    print(f"\n{'=' * 60}")
    print(f"DONE")
    print(f"  Clues saved:  {total_saved}")
    print(f"  Not found:    {total_not_found}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
