#!/usr/bin/env python3
"""Guardian Grid Backfill — fetch grid solutions from Guardian public API.

For each Guardian puzzle in our DB that lacks a grid (and is >= #21620 where
the API is available), fetches the JSON, extracts the grid solution, and
stores it in puzzle_grids.

Usage:
    python scraper/guardian/guardian_grid_backfill.py
    python scraper/guardian/guardian_grid_backfill.py --test    # 10 puzzles only
"""

import argparse
import sqlite3
import time

import requests

from guardian_all import build_grid_solution

DB_PATH = r"C:\Users\shute\PycharmProjects\cryptic_solver_V2\data\clues_master.db"

API_CUTOFF = 21620  # Puzzles below this return 404

TYPE_PATHS = ['cryptic', 'prize', 'everyman', 'quiptic', 'quick-cryptic']


def get_puzzles_needing_grids(conn):
    """Return sorted list of Guardian puzzle numbers >= API_CUTOFF without grids."""
    c = conn.cursor()
    c.execute("""
        SELECT DISTINCT p.puzzle_number
        FROM clues p
        LEFT JOIN puzzle_grids g ON p.source = g.source AND p.puzzle_number = g.puzzle_number
        WHERE p.source = 'guardian'
          AND g.source IS NULL
          AND CAST(p.puzzle_number AS INTEGER) >= ?
        ORDER BY CAST(p.puzzle_number AS INTEGER)
    """, (API_CUTOFF,))
    return [row[0] for row in c.fetchall()]


def fetch_grid(puzzle_number):
    """Try each type path until one returns 200 with grid data.

    Returns (solution, rows, cols, url) or (None, None, None, None).
    """
    for type_path in TYPE_PATHS:
        url = f"https://www.theguardian.com/crosswords/{type_path}/{puzzle_number}.json"
        try:
            r = requests.get(url, timeout=15)
        except Exception as e:
            print(f"    {type_path}: request error: {e}")
            continue

        if r.status_code != 200:
            continue

        try:
            data = r.json()
            solution, rows, cols = build_grid_solution(data)
            if solution:
                return solution, rows, cols, url
        except Exception as e:
            print(f"    {type_path}: parse error: {e}")
            continue

    return None, None, None, None


def backfill(test=False):
    conn = sqlite3.connect(DB_PATH)

    # Ensure puzzle_grids table exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS puzzle_grids (
            source TEXT NOT NULL,
            puzzle_number TEXT NOT NULL,
            solution TEXT,
            grid_rows INTEGER NOT NULL DEFAULT 15,
            grid_cols INTEGER NOT NULL DEFAULT 15,
            api_folder TEXT,
            api_type TEXT,
            api_id TEXT,
            PRIMARY KEY (source, puzzle_number)
        )
    """)

    puzzles = get_puzzles_needing_grids(conn)
    total = len(puzzles)
    print(f"Guardian puzzles needing grids: {total}")

    if test:
        puzzles = puzzles[-10:]  # Last 10 (most recent)
        print(f"TEST MODE: processing {len(puzzles)} puzzles only")

    succeeded = 0
    failed = 0
    failed_nums = []

    for i, pnum in enumerate(puzzles):
        solution, rows, cols, url = fetch_grid(pnum)

        if solution:
            # api_folder stores the source URL for Guardian puzzles
            conn.execute("""
                INSERT OR REPLACE INTO puzzle_grids
                (source, puzzle_number, solution, grid_rows, grid_cols, api_folder)
                VALUES (?, ?, ?, ?, ?, ?)
            """, ('guardian', pnum, solution, rows, cols, url))
            conn.commit()
            succeeded += 1
            print(f"  #{pnum}: OK ({rows}x{cols})")
        else:
            failed += 1
            failed_nums.append(int(pnum))
            print(f"  #{pnum}: FAILED (404 on all type paths)")

        # Progress every 100
        if (i + 1) % 100 == 0:
            print(f"  --- Progress: {i + 1}/{len(puzzles)}, {succeeded} OK, {failed} failed ---")

        # Rate limit: ~2 requests per second
        time.sleep(0.5)

    conn.close()

    print(f"\n{'=' * 60}")
    print(f"COMPLETE")
    print(f"{'=' * 60}")
    print(f"Succeeded: {succeeded}")
    print(f"Failed: {failed}")

    if failed_nums:
        print(f"\nFailed puzzle numbers ({len(failed_nums)}):")
        print(f"  {failed_nums[:50]}")
        if len(failed_nums) > 50:
            print(f"  ... and {len(failed_nums) - 50} more")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Guardian Grid Backfill")
    parser.add_argument("--test", action="store_true", help="Process 10 puzzles only")
    args = parser.parse_args()

    backfill(test=args.test)
