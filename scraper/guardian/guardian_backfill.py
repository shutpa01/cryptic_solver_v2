#!/usr/bin/env python3
"""Guardian Puzzle Backfill - One-time script to download historical puzzles.
Downloads cryptic, everyman, and genius puzzles going back as far as possible.
"""

import requests
import json
import sqlite3
import os
import re
from datetime import datetime, date, timedelta
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv('DB_PATH',
                    r"C:\Users\shute\PycharmProjects\AI_Solver\data\clues_master.db")

# Puzzle configurations
PUZZLES = {
    'cryptic': {
        'name': 'Guardian Cryptic',
        'url_pattern': 'https://www.theguardian.com/crosswords/cryptic/{number}.json',
        'current_number': 29911,  # As of 2026-01-22
    },
    'everyman': {
        'name': 'Everyman',
        'url_pattern': 'https://www.theguardian.com/crosswords/everyman/{number}.json',
        'current_number': 4096,  # Last one on Guardian before move to Observer
    },
    'genius': {
        'name': 'Genius',
        'url_pattern': 'https://www.theguardian.com/crosswords/genius/{number}.json',
        'current_number': 271,  # As of 2026-01-05
    }
}


def init_database():
    """Ensure database and table exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS guardian_clues (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            puzzle_type TEXT,
            puzzle_number TEXT,
            puzzle_date TEXT,
            setter TEXT,
            clue_number TEXT,
            direction TEXT,
            clue_text TEXT,
            enumeration TEXT,
            answer TEXT,
            explanation TEXT,
            published INTEGER DEFAULT 0,
            fetched_at TEXT
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_guardian_puzzle 
        ON guardian_clues(puzzle_type, puzzle_number)
    """)

    conn.commit()
    conn.close()


def puzzle_exists(puzzle_type, puzzle_number):
    """Check if puzzle already in database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT COUNT(*) FROM guardian_clues 
        WHERE puzzle_type = ? AND puzzle_number = ?
    """, (puzzle_type, str(puzzle_number)))
    count = cursor.fetchone()[0]
    conn.close()
    return count > 0


def fetch_puzzle(puzzle_type, puzzle_number):
    """Fetch puzzle JSON from Guardian API."""
    url = PUZZLES[puzzle_type]['url_pattern'].format(number=puzzle_number)

    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            return None
        else:
            print(f"  HTTP {response.status_code}")
            return None
    except Exception as e:
        print(f"  Error: {e}")
        return None


def parse_and_save(data, puzzle_type, puzzle_number):
    """Parse puzzle data and save to database."""
    crossword = data.get('crossword', {})

    puzzle_date = crossword.get('date', '')
    if isinstance(puzzle_date, int):
        puzzle_date = datetime.utcfromtimestamp(puzzle_date / 1000).strftime(
            '%A, %d %B %Y')

    setter = crossword.get('creator', {}).get('name', '')
    entries = crossword.get('entries', [])
    fetched_at = datetime.now().isoformat()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    clue_count = 0
    for entry in entries:
        clue_text = entry.get('clue', '')
        clue_text = re.sub(r'<[^>]+>', '', clue_text)

        enum_match = re.search(r'\(([0-9,\-\s]+)\)\s*$', clue_text)
        enumeration = enum_match.group(1) if enum_match else str(entry.get('length', ''))
        clean_text = re.sub(r'\s*\([0-9,\-\s]+\)\s*$', '', clue_text).strip()

        direction = entry.get('direction', '').lower()

        cursor.execute("""
            INSERT INTO guardian_clues 
            (puzzle_type, puzzle_number, puzzle_date, setter, clue_number, direction, 
             clue_text, enumeration, answer, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            puzzle_type,
            str(puzzle_number),
            puzzle_date,
            setter,
            str(entry.get('number', '')),
            direction,
            clean_text,
            enumeration,
            entry.get('solution', ''),
            fetched_at
        ))
        clue_count += 1

    conn.commit()
    conn.close()

    return clue_count, setter, puzzle_date


def backfill_puzzle_type(puzzle_type, start_number=None):
    """Download all puzzles of a type, going backwards from current."""
    config = PUZZLES[puzzle_type]

    if start_number is None:
        start_number = config['current_number']

    print(f"\n{'=' * 60}")
    print(f"BACKFILLING: {config['name']}")
    print(f"Starting from #{start_number}, going backwards")
    print(f"{'=' * 60}")

    consecutive_failures = 0
    max_failures = 10  # Stop after 10 consecutive 404s
    total_saved = 0
    total_skipped = 0
    current = start_number

    while consecutive_failures < max_failures and current > 0:
        # Check if already exists
        if puzzle_exists(puzzle_type, current):
            print(f"  #{current}: already in database")
            total_skipped += 1
            current -= 1
            consecutive_failures = 0  # Reset on skip (puzzle exists means valid number)
            continue

        # Fetch puzzle
        data = fetch_puzzle(puzzle_type, current)

        if data is None:
            consecutive_failures += 1
            print(f"  #{current}: not found ({consecutive_failures}/{max_failures})")
            current -= 1
            continue

        # Parse and save
        consecutive_failures = 0
        clue_count, setter, puzzle_date = parse_and_save(data, puzzle_type, current)
        total_saved += 1

        print(f"  #{current}: {setter or 'Unknown'} ({puzzle_date}) - {clue_count} clues")

        current -= 1

    print(f"\n{config['name']} complete:")
    print(f"  Saved: {total_saved}")
    print(f"  Skipped (already had): {total_skipped}")
    print(f"  Stopped at: #{current + 1}")

    return total_saved, total_skipped


def main():
    print("=" * 60)
    print("GUARDIAN PUZZLE BACKFILL")
    print(f"Date: {date.today().strftime('%A, %d %B %Y')}")
    print("=" * 60)

    init_database()

    # Check current counts
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT puzzle_type, COUNT(DISTINCT puzzle_number) 
        FROM guardian_clues 
        GROUP BY puzzle_type
    """)
    existing = dict(cursor.fetchall())
    conn.close()

    print("\nExisting puzzles in database:")
    for pt in PUZZLES:
        count = existing.get(pt, 0)
        print(f"  {pt}: {count}")

    grand_total_saved = 0
    grand_total_skipped = 0

    # Backfill each puzzle type
    for puzzle_type in ['cryptic', 'everyman', 'genius']:
        saved, skipped = backfill_puzzle_type(puzzle_type)
        grand_total_saved += saved
        grand_total_skipped += skipped

    print("\n" + "=" * 60)
    print("BACKFILL COMPLETE")
    print(f"Total saved: {grand_total_saved}")
    print(f"Total skipped: {grand_total_skipped}")
    print("=" * 60)


if __name__ == "__main__":
    main()