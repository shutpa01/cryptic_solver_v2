#!/usr/bin/env python3
"""Telegraph Backfill Processor
Reads telegraph_api_mapping.json (from backfill harvest) and saves puzzles to database.
Uses existing functions from telegraph_all.py.

This is separate from telegraph_harvest_processor.py which handles daily harvests.
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
HARVEST_FILE = SCRIPT_DIR / "telegraph_api_mapping.json"


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
    print("TELEGRAPH BACKFILL PROCESSOR")
    print("=" * 60)

    harvest = load_harvest()
    if not harvest:
        return

    # Count total puzzles
    total_puzzles = sum(len(puzzles) for puzzles in harvest.values())
    print(f"Puzzle types in harvest: {list(harvest.keys())}")
    print(f"Total puzzles in harvest: {total_puzzles}")
    print(f"Database: {DB_PATH}")

    stats = {'skipped': 0, 'fetched': 0, 'failed': 0}

    # Process each puzzle type
    for harvest_type, puzzles in harvest.items():
        puzzle_type = map_puzzle_type(harvest_type)
        print(f"\n{'='*60}")
        print(f"Processing: {harvest_type} ({len(puzzles)} puzzles)")
        print(f"{'='*60}")

        for puzzle_number, puzzle_info in puzzles.items():
            api_id = puzzle_info['api_id']
            folder = puzzle_info['folder']

            print(f"\n  #{puzzle_number} (API ID: {api_id})")

            # Check if already in database
            if puzzle_already_fetched(puzzle_type, puzzle_number):
                print(f"    Already in database - skipping")
                stats['skipped'] += 1
                continue

            # Fetch from API
            data = fetch_by_api_id(folder, harvest_type, api_id)
            if not data:
                stats['failed'] += 1
                continue

            # Parse and save
            parsed = parse_puzzle(data, puzzle_type)

            print(f"    Title: {parsed.get('title', 'N/A')}")
            print(f"    Clues: {len(parsed.get('across', []))} across, {len(parsed.get('down', []))} down")

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