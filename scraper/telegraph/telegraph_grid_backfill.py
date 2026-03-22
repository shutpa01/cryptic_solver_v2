"""Backfill Telegraph grid JSON files.

Sweeps through API ID ranges and saves grid JSON for all puzzles.
This gives us the cell-to-clue mapping needed for interactive solving.

Usage:
    python scraper/telegraph/telegraph_grid_backfill.py [--start ID] [--end ID] [--dry-run]

Defaults to scanning ID range 80100-86000 across all puzzle types.
Skips IDs where we already have a JSON file.
"""

import argparse
import json
import re
import sqlite3
import time
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH = SCRIPT_DIR.parent.parent / "data" / "clues_master.db"

# All Telegraph puzzle type folders
PUZZLE_TYPES = [
    ("cryptic-crossword-1", "cryptic-crossword"),
    ("prize-cryptic", "prize-cryptic"),
    ("toughie-crossword", "toughie-crossword"),
    ("prize-toughie", "prize-toughie"),
]

BASE_URL = "https://puzzlesdata.telegraph.co.uk/puzzles"


def extract_puzzle_number(title):
    """Extract puzzle number from API title like 'Cryptic Crossword No 31128'."""
    m = re.search(r"No\.?\s*(\d+)", title)
    return m.group(1) if m else None


def existing_json_ids():
    """Return set of api_ids that already have JSON files saved."""
    existing = set()
    for f in SCRIPT_DIR.glob("telegraph_*.json"):
        # Parse: telegraph_{type}_{api_id}.json
        m = re.match(r"telegraph_.*_(\d+)\.json", f.name)
        if m:
            existing.add(int(m.group(1)))
    return existing


def backfill(start_id, end_id, dry_run=False):
    existing = existing_json_ids()
    print(f"Found {len(existing)} existing JSON files")
    print(f"Scanning API IDs {start_id} to {end_id}")

    conn = sqlite3.connect(str(DB_PATH))
    fetched = 0
    skipped = 0
    errors = 0
    not_found = 0

    for api_id in range(start_id, end_id + 1):
        if api_id in existing:
            skipped += 1
            continue

        # Try each puzzle type
        found = False
        for folder, ptype in PUZZLE_TYPES:
            url = f"{BASE_URL}/{folder}/{ptype}-{api_id}.json"

            try:
                r = requests.get(url, timeout=15)
            except Exception as e:
                print(f"  [{api_id}] Error: {e}")
                errors += 1
                continue

            if r.status_code == 200:
                data = r.json()
                copy = data.get("json", {}).get("copy", {})
                title = copy.get("title", "")
                puzzle_number = extract_puzzle_number(title)

                if dry_run:
                    print(f"  [{api_id}] {folder}/{ptype}: {title} -> #{puzzle_number}")
                else:
                    # Save JSON
                    json_path = SCRIPT_DIR / f"telegraph_{ptype}-{api_id}.json"
                    with open(json_path, "w") as f:
                        json.dump(data, f)

                    # Update puzzle_grids with api info
                    if puzzle_number:
                        conn.execute("""
                            INSERT OR REPLACE INTO puzzle_grids
                            (source, puzzle_number, api_folder, api_type, api_id,
                             grid_rows, grid_cols)
                            VALUES (?, ?, ?, ?, ?, 15, 15)
                        """, ("telegraph", puzzle_number, folder, ptype, str(api_id)))
                        conn.commit()

                    print(f"  [{api_id}] Saved: {ptype} #{puzzle_number} ({title})")

                fetched += 1
                found = True
                break  # Don't try other types for this ID

            if r.status_code != 404:
                errors += 1

        if not found:
            not_found += 1

        # Rate limit: ~2 requests per second
        time.sleep(0.5)

        # Progress every 100
        if (api_id - start_id) % 100 == 0 and api_id > start_id:
            total_tried = api_id - start_id
            print(f"  Progress: {total_tried} IDs scanned, {fetched} fetched, {not_found} not found")

    conn.close()
    print(f"\nDone: {fetched} fetched, {skipped} already had, {not_found} not found, {errors} errors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill Telegraph grid JSON files")
    parser.add_argument("--start", type=int, default=80100, help="Start API ID")
    parser.add_argument("--end", type=int, default=86000, help="End API ID")
    parser.add_argument("--dry-run", action="store_true", help="Don't save, just show what would be fetched")
    args = parser.parse_args()

    backfill(args.start, args.end, args.dry_run)
