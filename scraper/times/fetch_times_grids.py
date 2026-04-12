#!/usr/bin/env python3
"""Fetch Times grids from harvested API URLs.

Reads URLs from times_api_mapping.json, fetches the JSON,
extracts grid solutions, and saves to puzzle_grids.
Only processes puzzles that exist in our clues DB and don't already have a grid.
"""

import json
import re
import sqlite3
import time
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH = SCRIPT_DIR.parent.parent / "data" / "clues_master.db"
HARVEST_FILE = SCRIPT_DIR / "times_api_mapping.json"


def main():
    if not HARVEST_FILE.exists():
        print(f"Harvest file not found: {HARVEST_FILE}")
        return

    harvest = json.loads(HARVEST_FILE.read_text())
    conn = sqlite3.connect(str(DB_PATH))

    # Get times puzzles needing grids
    needing = set()
    for row in conn.execute("""
        SELECT DISTINCT c.puzzle_number
        FROM clues c
        LEFT JOIN puzzle_grids g ON c.source = g.source AND c.puzzle_number = g.puzzle_number
        WHERE c.source = 'times'
          AND (g.source IS NULL OR g.solution IS NULL)
    """):
        needing.add(row[0])

    print(f"Times puzzles needing grids: {len(needing)}")

    succeeded = 0
    failed = 0
    skipped = 0
    failed_list = []

    for harvest_type in ['cryptic', 'sunday-cryptic']:
        puzzles = harvest.get(harvest_type, {})
        print(f"\n{harvest_type}: {len(puzzles)} URLs")

        for pnum, url in sorted(puzzles.items(), key=lambda x: int(x[0])):
            if pnum not in needing:
                skipped += 1
                continue

            try:
                r = requests.get(url, timeout=15)
                if r.status_code != 200:
                    failed += 1
                    failed_list.append((pnum, f'http_{r.status_code}'))
                    time.sleep(0.3)
                    continue

                data = r.json()
                inner = data.get('data', data)
                copy = inner.get('copy', {})
                settings = copy.get('settings', {})
                solution = settings.get('solution', '')
                gridsize = copy.get('gridsize', {})
                grid_rows = int(gridsize.get('rows', 15))
                grid_cols = int(gridsize.get('cols', 15))

                if solution and len(solution) == grid_rows * grid_cols:
                    conn.execute("""
                        INSERT OR REPLACE INTO puzzle_grids
                        (source, puzzle_number, solution, grid_rows, grid_cols, api_folder)
                        VALUES ('times', ?, ?, ?, ?, ?)
                    """, (pnum, solution, grid_rows, grid_cols, url))
                    conn.commit()
                    needing.discard(pnum)
                    succeeded += 1
                else:
                    failed += 1
                    failed_list.append((pnum, 'no_solution'))

            except Exception as e:
                failed += 1
                failed_list.append((pnum, str(e)[:40]))

            if (succeeded + failed) % 100 == 0 and (succeeded + failed) > 0:
                print(f"  --- {succeeded + failed + skipped} processed, {succeeded} OK, {failed} failed ---")
            time.sleep(0.3)

    conn.close()

    print(f"\n{'=' * 60}")
    print(f"Succeeded: {succeeded}")
    print(f"Skipped (already had grid): {skipped}")
    print(f"Failed: {failed}")
    print(f"Still needing: {len(needing)}")

    if failed_list:
        from collections import Counter
        by_reason = Counter(r for _, r in failed_list)
        print(f"\nFailures by reason:")
        for reason, cnt in by_reason.most_common():
            print(f"  {reason}: {cnt}")


if __name__ == "__main__":
    main()
