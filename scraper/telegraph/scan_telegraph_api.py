#!/usr/bin/env python3
"""Scan Telegraph API IDs to find grids for puzzles missing from the harvest.

The Telegraph puzzles API uses sequential api_ids. By scanning a range,
we can find puzzle_number -> api_id mappings and extract grid solutions
for puzzles that were missed by the date-picker harvester.

Only saves grids for puzzles that exist in our clues DB and don't already
have a grid solution.
"""

import re
import sqlite3
import time
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH = SCRIPT_DIR.parent.parent / "data" / "clues_master.db"

API_BASE = "https://puzzlesdata.telegraph.co.uk/puzzles/cryptic-crossword-1/cryptic-crossword-{}.json"


def main():
    conn = sqlite3.connect(str(DB_PATH))

    # Get telegraph puzzles that need grids
    needing = set()
    for row in conn.execute("""
        SELECT DISTINCT c.puzzle_number
        FROM clues c
        LEFT JOIN puzzle_grids g ON c.source = g.source AND c.puzzle_number = g.puzzle_number
        WHERE c.source = 'telegraph'
          AND (g.source IS NULL OR g.solution IS NULL)
    """):
        needing.add(str(row[0]))

    print(f"Telegraph puzzles needing grids: {len(needing)}")

    # Scan api_ids from 37195 (puzzle #27738, Mar 2015) upward
    # Stop when we've gone past all known puzzles or hit too many 404s
    start_aid = 37195
    end_aid = 39000  # well past the latest known gap
    consecutive_404 = 0
    max_consecutive_404 = 20

    saved = 0
    scanned = 0
    skipped = 0

    for api_id in range(start_aid, end_aid + 1):
        url = API_BASE.format(api_id)
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 404:
                consecutive_404 += 1
                if consecutive_404 >= max_consecutive_404:
                    print(f"\n{max_consecutive_404} consecutive 404s at api_id={api_id}, stopping")
                    break
                continue
            elif r.status_code != 200:
                print(f"  api_id={api_id}: HTTP {r.status_code}")
                continue

            consecutive_404 = 0
            data = r.json()
            copy = data.get('json', {}).get('copy', {})
            title = copy.get('title', '')

            # Extract puzzle number from title like "Cryptic Crossword No 27738"
            m = re.search(r'No\s+(\d+)', title)
            if not m:
                scanned += 1
                continue

            puzzle_num = m.group(1)

            if puzzle_num not in needing:
                skipped += 1
                scanned += 1
                continue

            # Extract grid solution
            settings = copy.get('settings', {})
            solution = settings.get('solution', '')
            gridsize = copy.get('gridsize', {})
            grid_rows = int(gridsize.get('rows', 15))
            grid_cols = int(gridsize.get('cols', 15))

            if solution and len(solution) == grid_rows * grid_cols:
                conn.execute("""
                    INSERT OR REPLACE INTO puzzle_grids
                    (source, puzzle_number, solution, grid_rows, grid_cols, api_folder, api_id)
                    VALUES ('telegraph', ?, ?, ?, ?, ?, ?)
                """, (puzzle_num, solution, grid_rows, grid_cols, url, str(api_id)))
                conn.commit()
                needing.discard(puzzle_num)
                saved += 1

            scanned += 1
            if scanned % 100 == 0:
                print(f"  --- api_id={api_id}: scanned {scanned}, saved {saved}, "
                      f"skipped {skipped}, still need {len(needing)} ---")

        except Exception as e:
            print(f"  api_id={api_id}: error: {e}")

        time.sleep(0.3)

    conn.close()

    print(f"\n{'=' * 60}")
    print(f"Scanned: {scanned}")
    print(f"Saved: {saved}")
    print(f"Skipped (not needed): {skipped}")
    print(f"Still needing grids: {len(needing)}")


if __name__ == "__main__":
    main()
