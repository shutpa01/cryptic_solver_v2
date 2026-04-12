#!/usr/bin/env python3
"""Save telegraph API URLs from harvest JSON into puzzle_grids.

Inserts rows with api_folder (URL) and api_id but NULL solution,
so URLs are preserved permanently even before grids are fetched.

Only processes cryptic-crossword and prize-cryptic (not toughie).
Skips puzzles that already have a puzzle_grids entry.
"""

import json
import sqlite3
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH = SCRIPT_DIR.parent.parent / "data" / "clues_master.db"
HARVEST_FILE = SCRIPT_DIR / "telegraph_api_mapping.json"

# Only these types — no toughie
TYPES_TO_SAVE = {
    'cryptic-crossword': 'telegraph',
    'prize-cryptic': 'telegraph',
}


def main():
    harvest = json.loads(HARVEST_FILE.read_text())
    conn = sqlite3.connect(str(DB_PATH))

    # Get existing puzzle_grids entries for telegraph
    existing = set()
    for row in conn.execute(
        "SELECT source, puzzle_number FROM puzzle_grids WHERE source = 'telegraph'"
    ):
        existing.add((row[0], row[1]))

    print(f"Existing telegraph grid entries: {len(existing)}")

    inserted = 0
    skipped = 0

    for harvest_type, db_source in TYPES_TO_SAVE.items():
        puzzles = harvest.get(harvest_type, {})
        print(f"\n{harvest_type} -> {db_source}: {len(puzzles)} URLs in harvest")

        for pnum, info in sorted(puzzles.items(), key=lambda x: int(x[0])):
            if (db_source, pnum) in existing:
                skipped += 1
                continue

            url = info.get('url') if isinstance(info, dict) else info
            api_id = info.get('api_id') if isinstance(info, dict) else None

            if not url:
                continue

            conn.execute("""
                INSERT OR IGNORE INTO puzzle_grids
                (source, puzzle_number, solution, grid_rows, grid_cols, api_folder, api_id)
                VALUES (?, ?, NULL, 15, 15, ?, ?)
            """, (db_source, pnum, url, api_id))
            existing.add((db_source, pnum))
            inserted += 1

    conn.commit()
    conn.close()

    print(f"\nInserted: {inserted} URL-only rows")
    print(f"Skipped (already existed): {skipped}")


if __name__ == "__main__":
    main()
