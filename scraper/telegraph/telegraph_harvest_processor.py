#!/usr/bin/env python3
"""Telegraph Harvest Processor
Reads telegraph_harvest.json and saves new puzzles to database.
Uses existing functions from telegraph_all.py.
"""

import json
import requests
from pathlib import Path
from datetime import datetime

# Import from telegraph_all
from telegraph_all import (
    parse_puzzle,
    save_to_database,
    puzzle_already_fetched,
    DB_PATH
)

SCRIPT_DIR = Path(__file__).parent
HARVEST_FILE = SCRIPT_DIR / "telegraph_harvest.json"


def load_harvest():
    """Load the harvest JSON file."""
    if not HARVEST_FILE.exists():
        print(f"Harvest file not found: {HARVEST_FILE}")
        return None

    with open(HARVEST_FILE) as f:
        return json.load(f)


def fetch_by_api_id(folder, puzzle_type, api_id):
    """Fetch puzzle data using the exact API ID from harvest."""
    url = f"https://puzzlesdata.telegraph.co.uk/puzzles/{folder}/{puzzle_type}-{api_id}.json"
    print(f"  Fetching: {url}")

    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"  Error: HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"  Error: {e}")
        return None


def map_puzzle_type(harvest_type):
    """Map harvest type to telegraph_all puzzle_type key."""
    # The harvest uses types like "cryptic-crossword", "prize-cryptic", etc.
    # telegraph_all uses keys like "cryptic", "prize-cryptic", etc.
    mapping = {
        'cryptic-crossword': 'cryptic',
        'prize-cryptic': 'prize-cryptic',
        'prize-toughie': 'prize-toughie',
        'toughie-crossword': 'toughie',
    }
    return mapping.get(harvest_type, harvest_type)


def process_harvest():
    """Process all puzzles in the harvest file."""
    print("=" * 60)
    print("TELEGRAPH HARVEST PROCESSOR")
    print("=" * 60)

    harvest = load_harvest()
    if not harvest:
        return

    puzzles = harvest.get('puzzles', [])
    print(f"Harvest date: {harvest.get('harvested_at')}")
    print(f"Puzzles in harvest: {len(puzzles)}")
    print(f"Database: {DB_PATH}")

    stats = {'skipped': 0, 'fetched': 0, 'failed': 0}

    for puzzle in puzzles:
        folder = puzzle['folder']
        harvest_type = puzzle['type']
        api_id = puzzle['api_id']
        puzzle_number = puzzle['puzzle_number']
        puzzle_date = puzzle.get('date', '')

        # Map to telegraph_all puzzle type
        puzzle_type = map_puzzle_type(harvest_type)

        print(f"\n{puzzle['name']} #{puzzle_number} ({puzzle_date})")
        print(f"  Type: {puzzle_type} | API ID: {api_id}")

        # Check if already in database
        if puzzle_already_fetched(puzzle_type, puzzle_number):
            print(f"  Already in database - skipping")
            stats['skipped'] += 1
            continue

        # Fetch from API
        data = fetch_by_api_id(folder, harvest_type, api_id)
        if not data:
            stats['failed'] += 1
            continue

        # Parse and save
        parsed = parse_puzzle(data, puzzle_type)

        print(f"  Title: {parsed.get('title')}")
        print(
            f"  Clues: {len(parsed.get('across', []))} across, {len(parsed.get('down', []))} down")

        clue_count = save_to_database(parsed, puzzle_type)
        if clue_count > 0:
            stats['fetched'] += 1
        else:
            stats['failed'] += 1

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Skipped (already in DB): {stats['skipped']}")
    print(f"Fetched and saved:       {stats['fetched']}")
    print(f"Failed:                  {stats['failed']}")

    return stats


if __name__ == "__main__":
    process_harvest()