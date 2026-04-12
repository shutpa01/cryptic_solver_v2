#!/usr/bin/env python3
"""Match orphan puzzle_grids entries to legacy-numbered clues.

The cryptic-crossword January harvest created grid entries under current
Telegraph numbering (e.g. #29923), but our clues use legacy numbering
(e.g. #28624). This script matches them by fetching the API JSON and
comparing 1-across clue text, then copies the grid to the legacy number.
"""

import re
import sqlite3
import time
from html import unescape
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH = SCRIPT_DIR.parent.parent / "data" / "clues_master.db"


def clean_clue(text):
    """Normalise clue text for comparison."""
    text = unescape(text)
    text = re.sub(r'<[^>]+>', '', text)        # strip HTML tags
    text = re.sub(r'[^\w\s]', '', text)        # strip punctuation
    text = text.lower().strip()
    return ' '.join(text.split())


def main():
    conn = sqlite3.connect(str(DB_PATH))

    # Find orphan grid entries: have puzzle_grids row but no matching clues
    orphans = conn.execute("""
        SELECT g.puzzle_number, g.solution, g.grid_rows, g.grid_cols, g.api_folder, g.api_id
        FROM puzzle_grids g
        LEFT JOIN clues c ON g.source = c.source AND g.puzzle_number = c.puzzle_number
        WHERE g.source = 'telegraph'
          AND g.solution IS NOT NULL
          AND c.source IS NULL
        ORDER BY CAST(g.puzzle_number AS INTEGER)
    """).fetchall()

    print(f"Orphan grid entries (no matching clues): {len(orphans)}")

    matched = 0
    failed = 0
    orphans_to_remove = []

    for pnum, solution, rows, cols, url, api_id in orphans:
        if not url or not url.startswith('http'):
            print(f"  #{pnum}: no valid URL, skipping")
            failed += 1
            continue

        try:
            r = requests.get(url, timeout=15)
            if r.status_code != 200:
                print(f"  #{pnum}: HTTP {r.status_code}")
                failed += 1
                time.sleep(0.5)
                continue

            data = r.json()
            copy = data.get('json', {}).get('copy', {})
            clues_data = copy.get('clues', [])

            # Extract 1-across clue text
            first_clue = None
            for section in clues_data:
                if section.get('title', '').lower() == 'across':
                    for clue in section.get('clues', []):
                        if clue.get('number') == 1:
                            first_clue = clue.get('clue', '')
                            break
                    break

            if not first_clue:
                print(f"  #{pnum}: no 1-across clue in API JSON")
                failed += 1
                time.sleep(0.5)
                continue

            cleaned = clean_clue(first_clue)

            # Search for matching legacy puzzle by 1-across text
            candidates = conn.execute("""
                SELECT puzzle_number, clue_text
                FROM clues
                WHERE source = 'telegraph'
                  AND (clue_number = '1' OR clue_number = '1.')
                  AND direction = 'across'
            """).fetchall()

            match_pnum = None
            for cand_pnum, cand_text in candidates:
                if clean_clue(cand_text) == cleaned:
                    match_pnum = cand_pnum
                    break

            if not match_pnum:
                print(f"  #{pnum}: no clue match for '{first_clue[:50]}...'")
                failed += 1
            else:
                # Copy grid to legacy puzzle number
                conn.execute("""
                    INSERT OR REPLACE INTO puzzle_grids
                    (source, puzzle_number, solution, grid_rows, grid_cols, api_folder, api_id)
                    VALUES ('telegraph', ?, ?, ?, ?, ?, ?)
                """, (match_pnum, solution, rows, cols, url, api_id))
                conn.commit()
                orphans_to_remove.append(pnum)
                matched += 1
                print(f"  #{pnum} -> #{match_pnum}: matched")

        except Exception as e:
            print(f"  #{pnum}: error: {e}")
            failed += 1

        time.sleep(0.5)

    # Remove orphan entries that were successfully matched
    for pnum in orphans_to_remove:
        conn.execute(
            "DELETE FROM puzzle_grids WHERE source = 'telegraph' AND puzzle_number = ?",
            (pnum,)
        )
    conn.commit()
    conn.close()

    print(f"\nMatched: {matched}")
    print(f"Failed: {failed}")
    if orphans_to_remove:
        print(f"Removed {len(orphans_to_remove)} orphan entries")


if __name__ == "__main__":
    main()
