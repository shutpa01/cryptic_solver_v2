#!/usr/bin/env python3
"""Guardian Puzzle Scraper - All Puzzle Types
Fetches puzzles from Guardian JSON API and saves to database.
"""

import requests
import json
import sqlite3
import os
import sys
import re
from datetime import datetime, date, timedelta
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv('DB_PATH',
                    r"C:\Users\shute\PycharmProjects\AI_Solver\data\clues_master.db")

# Guardian puzzle types
PUZZLE_TYPES = {
    'cryptic': {
        'name': 'Guardian Cryptic',
        'url_path': 'cryptic',
        'reference_date': date(2026, 1, 22),  # Thursday
        'reference_number': 29911,
        'days': [0, 1, 2, 3, 4],  # Mon-Fri
        'source': 'guardian-cryptic'
    },
    'quick-cryptic': {
        'name': 'Quick Cryptic',
        'url_path': 'quick-cryptic',
        'reference_date': date(2026, 1, 18),  # Saturday
        'reference_number': 94,  # From user's example
        'days': [5],  # Saturday only
        'source': 'guardian-quick-cryptic'
    },
    'quick': {
        'name': 'Guardian Quick',
        'url_path': 'quick',
        'reference_date': date(2026, 1, 23),
        'reference_number': 17000,  # Placeholder - need to verify
        'days': [0, 1, 2, 3, 4, 5],  # Mon-Sat
        'source': 'guardian-quick'
    },
    'prize': {
        'name': 'Guardian Prize',
        'url_path': 'prize',
        'reference_date': date(2026, 1, 18),  # Saturday
        'reference_number': 29000,  # Placeholder
        'days': [5],  # Saturday only
        'source': 'guardian-prize'
    },
    'everyman': {
        'name': 'Everyman',
        'url_path': 'everyman',
        'reference_date': date(2026, 1, 19),  # Sunday
        'reference_number': 4050,  # Placeholder
        'days': [6],  # Sunday only
        'source': 'guardian-everyman'
    },
    'speedy': {
        'name': 'Speedy',
        'url_path': 'speedy',
        'reference_date': date(2026, 1, 19),  # Sunday
        'reference_number': 1500,  # Placeholder
        'days': [6],  # Sunday only
        'source': 'guardian-speedy'
    },
    'quiptic': {
        'name': 'Quiptic',
        'url_path': 'quiptic',
        'reference_date': date(2026, 1, 19),  # Sunday
        'reference_number': 1300,  # Placeholder
        'days': [6],  # Sunday only
        'source': 'guardian-quiptic'
    }
}


def count_matching_days(start_date, end_date, weekdays):
    """Count how many days between start and end fall on specified weekdays."""
    count = 0
    step = 1 if end_date >= start_date else -1
    current = start_date

    while current != end_date:
        current += timedelta(days=step)
        if current.weekday() in weekdays:
            count += step

    return count


def get_puzzle_number_for_date(puzzle_type, target_date):
    """Calculate puzzle number for a given date."""
    config = PUZZLE_TYPES[puzzle_type]
    ref_date = config['reference_date']
    ref_number = config['reference_number']
    weekdays = config['days']

    diff = count_matching_days(ref_date, target_date, weekdays)
    return ref_number + diff


def get_puzzle_data(puzzle_type, puzzle_number):
    """Fetch puzzle data from Guardian API."""
    url_path = PUZZLE_TYPES[puzzle_type]['url_path']
    url = f"https://www.theguardian.com/crosswords/{url_path}/{puzzle_number}.json"
    print(f"Fetching: {url}")

    response = requests.get(url, timeout=30)
    if response.status_code != 200:
        print(f"Error: HTTP {response.status_code}")
        return None

    return response.json()


def parse_puzzle(data, puzzle_type):
    """Parse the Guardian API response."""
    crossword = data.get('crossword', {})

    puzzle_number = crossword.get('number', 0)
    name = crossword.get('name', '')
    setter = crossword.get('creator', {}).get('name', '')
    puzzle_date = crossword.get('date', '')

    # Convert epoch to readable date if needed (use UTC for consistency)
    if isinstance(puzzle_date, int):
        puzzle_date = datetime.utcfromtimestamp(puzzle_date / 1000).strftime(
            '%A, %d %B %Y')

    entries = crossword.get('entries', [])

    across = []
    down = []

    for entry in entries:
        clue_text = entry.get('clue', '')
        # Remove HTML tags
        clue_text = re.sub(r'<[^>]+>', '', clue_text)

        # Extract enumeration from end of clue
        enum_match = re.search(r'\(([0-9,\-\s]+)\)\s*$', clue_text)
        enumeration = enum_match.group(1) if enum_match else str(entry.get('length', ''))

        # Remove enumeration from clue text
        clean_text = re.sub(r'\s*\([0-9,\-\s]+\)\s*$', '', clue_text).strip()

        clue_obj = {
            'number': entry.get('number', ''),
            'clue': clean_text,
            'answer': entry.get('solution', ''),
            'enumeration': enumeration
        }

        direction = entry.get('direction', '').lower()
        if direction == 'across':
            across.append(clue_obj)
        elif direction == 'down':
            down.append(clue_obj)

    return {
        'puzzle_type': puzzle_type,
        'puzzle_number': puzzle_number,
        'title': name,
        'setter': setter,
        'date': puzzle_date,
        'across': across,
        'down': down
    }


def puzzle_already_fetched(puzzle_type, puzzle_number):
    """Check if puzzle is already in database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Ensure table exists
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
        SELECT COUNT(*) FROM guardian_clues 
        WHERE puzzle_type = ? AND puzzle_number = ?
    """, (puzzle_type, str(puzzle_number)))

    count = cursor.fetchone()[0]
    conn.close()
    return count > 0


def save_to_database(puzzle_data, puzzle_type):
    """Save puzzle clues to guardian_clues table."""
    print(f"Saving to guardian_clues table...")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create table if not exists
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

    # Index for quick lookups
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_guardian_puzzle 
        ON guardian_clues(puzzle_type, puzzle_number)
    """)

    puzzle_number = puzzle_data.get('puzzle_number', 0)
    puzzle_date = puzzle_data.get('date', '')
    setter = puzzle_data.get('setter', '')
    fetched_at = datetime.now().isoformat()

    clue_count = 0
    for direction in ['across', 'down']:
        for clue in puzzle_data.get(direction, []):
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
                str(clue.get('number', '')),
                direction,
                clue.get('clue', ''),
                clue.get('enumeration', ''),
                clue.get('answer', ''),
                fetched_at
            ))
            clue_count += 1

    conn.commit()
    conn.close()

    print(f"Saved {clue_count} clues")
    return clue_count


def get_todays_puzzles():
    """Get list of puzzle types available today."""
    today = date.today()
    weekday = today.weekday()

    available = []
    for puzzle_type, config in PUZZLE_TYPES.items():
        if weekday in config['days']:
            available.append(puzzle_type)

    return available


def fetch_puzzle(puzzle_type, puzzle_number=None, target_date=None, force=False):
    """Fetch and save a single puzzle."""
    config = PUZZLE_TYPES[puzzle_type]

    if puzzle_number is None:
        if target_date is None:
            target_date = date.today()
        puzzle_number = get_puzzle_number_for_date(puzzle_type, target_date)

    # Check if already in database
    if not force and puzzle_already_fetched(puzzle_type, puzzle_number):
        print(f"\n{config['name']} #{puzzle_number} already in database - skipping")
        return None

    print(f"\n{'=' * 50}")
    print(f"{config['name']}")
    print(f"Puzzle #: {puzzle_number}")

    data = get_puzzle_data(puzzle_type, puzzle_number)
    if not data:
        return None

    puzzle = parse_puzzle(data, puzzle_type)

    print(f"Title: {puzzle.get('title')}")
    print(f"Setter: {puzzle.get('setter')}")
    print(f"Date: {puzzle.get('date')}")
    print(
        f"Clues: {len(puzzle.get('across', []))} across, {len(puzzle.get('down', []))} down")

    if puzzle.get('across'):
        first = puzzle['across'][0]
        print(f"Sample: {first['number']}. {first['clue'][:40]}... = {first['answer']}")

    save_to_database(puzzle, puzzle_type)

    json_path = f"guardian_{puzzle_type}_{puzzle.get('puzzle_number')}.json"
    with open(json_path, 'w') as f:
        json.dump(puzzle, f, indent=2)
    print(f"JSON: {json_path}")

    return puzzle


def main():
    print("=" * 60)
    print("GUARDIAN PUZZLE SCRAPER")
    print("=" * 60)

    today = date.today()
    print(f"Today: {today.strftime('%A, %d %B %Y')}")

    available = get_todays_puzzles()
    print(f"Available today: {', '.join(available) if available else 'None'}")

    if len(sys.argv) > 1:
        arg = sys.argv[1]

        if arg == '--all':
            for pt in available:
                fetch_puzzle(pt)

        elif arg == '--list':
            print("\nPuzzle Types:")
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            for pt, config in PUZZLE_TYPES.items():
                day_names = [days[d] for d in config['days']]
                print(f"  {pt:15} {config['name']:20} ({', '.join(day_names)})")

        elif arg == '--status':
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT puzzle_type, puzzle_number, puzzle_date, setter,
                       COUNT(*) as clues,
                       SUM(CASE WHEN explanation IS NOT NULL THEN 1 ELSE 0 END) as explained,
                       MAX(published) as published
                FROM guardian_clues
                GROUP BY puzzle_type, puzzle_number
                ORDER BY fetched_at DESC
                LIMIT 20
            """)
            rows = cursor.fetchall()
            conn.close()

            print("\nRecent puzzles in guardian_clues:")
            print(
                f"{'Type':12} {'#':>6} {'Setter':15} {'Clues':>6} {'Expl':>6} {'Pub':>4}")
            print("-" * 55)
            for row in rows:
                pub = "Yes" if row[6] else "No"
                setter = (row[3] or '')[:15]
                print(
                    f"{row[0]:12} {row[1]:>6} {setter:15} {row[4]:>6} {row[5]:>6} {pub:>4}")

        elif arg in PUZZLE_TYPES:
            puzzle_number = int(sys.argv[2]) if len(sys.argv) > 2 else None
            fetch_puzzle(arg, puzzle_number)

        else:
            try:
                puzzle_number = int(arg)
                fetch_puzzle('cryptic', puzzle_number)
            except ValueError:
                print(f"\nUsage:")
                print("  python guardian_all.py              # Today's available puzzles")
                print("  python guardian_all.py --all        # Same as above")
                print("  python guardian_all.py --list       # List puzzle types")
                print("  python guardian_all.py --status     # Show puzzles in database")
                print("  python guardian_all.py cryptic      # Today's cryptic")
                print("  python guardian_all.py cryptic 29347  # Specific puzzle number")
    else:
        # Default: fetch all puzzles available today
        for pt in available:
            fetch_puzzle(pt)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()