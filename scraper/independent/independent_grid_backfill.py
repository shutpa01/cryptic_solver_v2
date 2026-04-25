#!/usr/bin/env python3
"""Independent Grid Backfill — fetch grids from edition.independent.co.uk
for puzzles that have clues but no stored grid.

Uses the existing edition scraper chain:
    date → edition ID → content.xml → puzzle URL → puzzle HTML → JSON → grid

Usage:
    python scraper/independent/independent_grid_backfill.py
    python scraper/independent/independent_grid_backfill.py --test    # 10 puzzles only
"""

import argparse
import sqlite3
import time
from datetime import date, datetime
from pathlib import Path

from independent_edition import (
    fetch_content_xml,
    find_cryptic_puzzle_url,
    fetch_puzzle_json,
    parse_puzzle,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH = SCRIPT_DIR.parent.parent / "data" / "clues_master.db"


def get_puzzles_needing_grids(conn):
    """Return list of (puzzle_number, publication_date) for Independent puzzles without grids."""
    c = conn.cursor()
    c.execute("""
        SELECT DISTINCT p.puzzle_number, MIN(c.publication_date)
        FROM clues c
        JOIN (
            SELECT DISTINCT source, puzzle_number FROM clues WHERE source = 'independent'
        ) p ON c.source = p.source AND c.puzzle_number = p.puzzle_number
        LEFT JOIN puzzle_grids g ON p.source = g.source AND p.puzzle_number = g.puzzle_number
        WHERE p.source = 'independent'
          AND g.source IS NULL
          AND c.publication_date IS NOT NULL
        GROUP BY p.puzzle_number
        ORDER BY c.publication_date
    """)
    return c.fetchall()


def save_grid(conn, puzzle_number, solution, rows, cols, url):
    """Save grid to puzzle_grids. api_folder stores the source URL."""
    conn.execute("""
        INSERT OR REPLACE INTO puzzle_grids
        (source, puzzle_number, solution, grid_rows, grid_cols, api_folder)
        VALUES (?, ?, ?, ?, ?, ?)
    """, ('independent', str(puzzle_number), solution, rows, cols, url))
    conn.commit()


def backfill(test=False):
    conn = sqlite3.connect(str(DB_PATH))

    puzzles = get_puzzles_needing_grids(conn)
    print(f"Independent puzzles needing grids: {len(puzzles)}")

    if test:
        puzzles = puzzles[-10:]
        print(f"TEST MODE: processing {len(puzzles)} puzzles only")

    succeeded = 0
    failed = 0
    failed_list = []

    for i, (pnum, pub_date_str) in enumerate(puzzles):
        try:
            puzzle_date = datetime.strptime(pub_date_str, '%Y-%m-%d').date()
        except (ValueError, TypeError):
            print(f"  #{pnum}: bad date '{pub_date_str}'")
            failed += 1
            failed_list.append((pnum, 'bad_date'))
            continue

        # Step 1: Fetch content.xml for this date
        xml_root = fetch_content_xml(puzzle_date)
        if xml_root is None:
            print(f"  #{pnum} ({pub_date_str}): no content.xml")
            failed += 1
            failed_list.append((pnum, 'no_content_xml'))
            time.sleep(0.5)
            continue

        # Step 2: Find the cryptic puzzle URL
        puzzle_url, puzzle_title = find_cryptic_puzzle_url(xml_root)
        if puzzle_url is None:
            print(f"  #{pnum} ({pub_date_str}): no cryptic puzzle in edition")
            failed += 1
            failed_list.append((pnum, 'no_cryptic'))
            time.sleep(0.5)
            continue

        # Step 3: Fetch puzzle JSON
        puzzle_json = fetch_puzzle_json(puzzle_url)
        if puzzle_json is None:
            print(f"  #{pnum} ({pub_date_str}): failed to fetch puzzle JSON")
            failed += 1
            failed_list.append((pnum, 'no_json'))
            time.sleep(0.5)
            continue

        # Step 4: Parse and extract grid
        try:
            clues, grid_solution, grid_rows, grid_cols, api_pnum = parse_puzzle(
                puzzle_json, puzzle_date, puzzle_title)
        except Exception as e:
            print(f"  #{pnum} ({pub_date_str}): parse error: {e}")
            failed += 1
            failed_list.append((pnum, 'parse_error'))
            time.sleep(0.5)
            continue

        if grid_solution and len(grid_solution) == grid_rows * grid_cols:
            save_grid(conn, pnum, grid_solution, grid_rows, grid_cols, puzzle_url)
            succeeded += 1
            print(f"  #{pnum} ({pub_date_str}): OK ({grid_rows}x{grid_cols})")
        else:
            print(f"  #{pnum} ({pub_date_str}): no grid solution in data")
            failed += 1
            failed_list.append((pnum, 'no_solution'))

        if (i + 1) % 100 == 0:
            print(f"  --- Progress: {i + 1}/{len(puzzles)}, {succeeded} OK, {failed} failed ---")

        time.sleep(1)

    conn.close()

    print(f"\n{'=' * 60}")
    print("COMPLETE")
    print(f"{'=' * 60}")
    print(f"Succeeded: {succeeded}")
    print(f"Failed: {failed}")

    if failed_list:
        from collections import Counter
        by_reason = Counter(r for _, r in failed_list)
        print(f"\nFailures by reason:")
        for reason, cnt in by_reason.most_common():
            print(f"  {reason}: {cnt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Independent Grid Backfill")
    parser.add_argument("--test", action="store_true", help="Process 10 puzzles only")
    args = parser.parse_args()

    backfill(test=args.test)
