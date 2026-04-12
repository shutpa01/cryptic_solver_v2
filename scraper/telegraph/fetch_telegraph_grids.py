#!/usr/bin/env python3
"""Fetch grids for telegraph puzzle_grids rows that have a URL but no solution.

Reads api_folder (URL) from puzzle_grids where solution IS NULL,
fetches the JSON, extracts the grid solution, and updates the row.

Only processes source='telegraph' (not toughie).
"""

import sqlite3
import time
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH = SCRIPT_DIR.parent.parent / "data" / "clues_master.db"


def main():
    conn = sqlite3.connect(str(DB_PATH))

    rows = conn.execute("""
        SELECT puzzle_number, api_folder, api_id
        FROM puzzle_grids
        WHERE source = 'telegraph'
          AND solution IS NULL
          AND api_folder IS NOT NULL
        ORDER BY CAST(puzzle_number AS INTEGER)
    """).fetchall()

    print(f"Puzzles needing grids (have URL, no solution): {len(rows)}")

    succeeded = 0
    failed = 0
    failed_list = []

    for i, (pnum, url, api_id) in enumerate(rows, 1):
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
            grid_rows = int(gridsize.get('rows', 15))
            grid_cols = int(gridsize.get('cols', 15))

            if solution and len(solution) == grid_rows * grid_cols:
                conn.execute("""
                    UPDATE puzzle_grids
                    SET solution = ?, grid_rows = ?, grid_cols = ?
                    WHERE source = 'telegraph' AND puzzle_number = ?
                """, (solution, grid_rows, grid_cols, pnum))
                conn.commit()
                succeeded += 1
                if succeeded % 100 == 0:
                    print(f"  --- Progress: {i}/{len(rows)}, {succeeded} OK, {failed} failed ---")
            else:
                print(f"  #{pnum}: no solution in JSON (len={len(solution) if solution else 0})")
                failed += 1
                failed_list.append((pnum, 'no_solution'))

        except Exception as e:
            print(f"  #{pnum}: error: {e}")
            failed += 1
            failed_list.append((pnum, 'error'))

        time.sleep(0.5)

    conn.close()

    print(f"\n{'=' * 60}")
    print(f"Succeeded: {succeeded}")
    print(f"Failed: {failed}")

    if failed_list:
        from collections import Counter
        by_reason = Counter(r for _, r in failed_list)
        print(f"\nFailures by reason:")
        for reason, cnt in by_reason.most_common():
            print(f"  {reason}: {cnt}")


if __name__ == "__main__":
    main()
