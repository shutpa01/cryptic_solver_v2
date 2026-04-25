#!/usr/bin/env python3
"""Telegraph Grid Backfill — fetch grids using URLs from telegraph_api_mapping.json.

For each URL in the mapping file, fetches the JSON and extracts the grid solution.
Only processes puzzles that exist in our DB and don't already have a grid.

Usage:
    python scraper/telegraph/telegraph_grid_backfill_v2.py
    python scraper/telegraph/telegraph_grid_backfill_v2.py --test    # 10 puzzles only
"""

import argparse
import json
import sqlite3
import time
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH = SCRIPT_DIR.parent.parent / "data" / "clues_master.db"
HARVEST_FILE = SCRIPT_DIR / "telegraph_api_mapping.json"

# Map harvest puzzle types to DB source names
SOURCE_MAP = {
    'cryptic-crossword': 'telegraph',
    'toughie-crossword': 'telegraph-toughie',
    'prize-cryptic': 'telegraph',
    'prize-toughie': 'telegraph-toughie',
}


def get_puzzles_needing_grids(conn):
    """Return set of (source, puzzle_number) for Telegraph puzzles without grids."""
    c = conn.cursor()
    c.execute("""
        SELECT DISTINCT p.source, p.puzzle_number
        FROM clues p
        LEFT JOIN puzzle_grids g ON p.source = g.source AND p.puzzle_number = g.puzzle_number
        WHERE p.source IN ('telegraph', 'telegraph-toughie')
          AND g.source IS NULL
    """)
    return set((row[0], row[1]) for row in c.fetchall())


def save_grid(conn, source, puzzle_number, solution, rows, cols, url):
    """Save grid to puzzle_grids. api_folder stores the source URL."""
    conn.execute("""
        INSERT OR REPLACE INTO puzzle_grids
        (source, puzzle_number, solution, grid_rows, grid_cols, api_folder)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (source, str(puzzle_number), solution, rows, cols, url))
    conn.commit()


def backfill(test=False):
    if not HARVEST_FILE.exists():
        print(f"Harvest file not found: {HARVEST_FILE}")
        print("Run telegraph_backfill.py first to collect API URLs.")
        return

    harvest = json.loads(HARVEST_FILE.read_text())
    conn = sqlite3.connect(str(DB_PATH))

    needing = get_puzzles_needing_grids(conn)
    print(f"Telegraph/toughie puzzles needing grids: {len(needing)}")

    succeeded = 0
    failed = 0
    skipped = 0
    failed_list = []
    processed = 0

    for harvest_type, puzzles in harvest.items():
        if harvest_type == '_completed_months' or not isinstance(puzzles, dict):
            continue

        db_source = SOURCE_MAP.get(harvest_type)
        if not db_source:
            continue

        print(f"\n{'=' * 60}")
        print(f"{harvest_type} -> {db_source}: {len(puzzles)} URLs")
        print(f"{'=' * 60}")

        for pnum, info in sorted(puzzles.items(), key=lambda x: int(x[0])):
            if (db_source, pnum) not in needing:
                skipped += 1
                continue

            if test and processed >= 10:
                print(f"\nTEST MODE: stopping after {processed} puzzles")
                break

            url = info.get('url') if isinstance(info, dict) else info
            if not url:
                failed += 1
                failed_list.append((pnum, 'no_url'))
                continue

            try:
                r = requests.get(url, timeout=15)
                if r.status_code != 200:
                    print(f"  #{pnum}: HTTP {r.status_code}")
                    failed += 1
                    failed_list.append((pnum, f'http_{r.status_code}'))
                    time.sleep(0.5)
                    continue

                data = r.json()
                copy = data.get('json', {}).get('copy', {})
                settings = copy.get('settings', {})
                solution = settings.get('solution', '')
                gridsize = copy.get('gridsize', {})
                rows = int(gridsize.get('rows', 15))
                cols = int(gridsize.get('cols', 15))

                if solution and len(solution) == rows * cols:
                    save_grid(conn, db_source, pnum, solution, rows, cols, url)
                    needing.discard((db_source, pnum))
                    succeeded += 1
                    print(f"  #{pnum}: OK ({rows}x{cols})")
                else:
                    print(f"  #{pnum}: no solution in JSON")
                    failed += 1
                    failed_list.append((pnum, 'no_solution'))

            except Exception as e:
                print(f"  #{pnum}: error: {e}")
                failed += 1
                failed_list.append((pnum, 'error'))

            processed += 1
            if processed % 100 == 0:
                print(f"  --- Progress: {processed}, {succeeded} OK, {failed} failed ---")
            time.sleep(0.5)

    conn.close()

    print(f"\n{'=' * 60}")
    print("COMPLETE")
    print(f"{'=' * 60}")
    print(f"Succeeded: {succeeded}")
    print(f"Skipped (already had grid or not in DB): {skipped}")
    print(f"Failed: {failed}")

    if failed_list:
        from collections import Counter
        by_reason = Counter(r for _, r in failed_list)
        print(f"\nFailures by reason:")
        for reason, cnt in by_reason.most_common():
            print(f"  {reason}: {cnt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Telegraph Grid Backfill")
    parser.add_argument("--test", action="store_true", help="Process 10 puzzles only")
    args = parser.parse_args()

    backfill(test=args.test)
