#!/usr/bin/env python3
"""Telegraph Puzzle Scraper - All Puzzle Types
Supports: Daily Cryptic (Mon-Fri), Saturday Cryptic (Sat), Prize Cryptic (Sun), Toughie (Tue-Fri), Prize Toughie (Sun)
"""

import requests
import json
import sqlite3
import os
import sys
import re
from datetime import datetime, date, timedelta
from dotenv import load_dotenv
from html import unescape

load_dotenv()

DB_PATH = os.getenv('DB_PATH',
                    r"C:\Users\shute\PycharmProjects\AI_Solver\data\clues_master.db")

# Puzzle type configurations
PUZZLE_TYPES = {
    'cryptic': {
        'name': 'Daily Cryptic',
        'api_path': 'cryptic-crossword-1/cryptic-crossword',
        'offset': 50061,
        'reference_date': date(2026, 1, 23),  # Thursday
        'reference_number': 31144,
        'days': [0, 1, 2, 3, 4],  # Mon-Fri only
        'source': 'telegraph-cryptic'
    },
    'saturday-cryptic': {
        'name': 'Saturday Cryptic',
        'api_path': 'prize-cryptic/prize-cryptic',
        'offset': 50054,
        'reference_date': date(2026, 1, 24),  # Saturday
        'reference_number': 31145,
        'days': [5],  # Saturday only
        'source': 'telegraph-cryptic'
    },
    'prize-cryptic': {
        'name': 'Prize Cryptic',
        'api_path': 'prize-cryptic/prize-cryptic',
        'offset': 77239,
        'reference_date': date(2026, 1, 18),  # Sunday
        'reference_number': 3352,
        'days': [6],  # Sunday only
        'source': 'telegraph-prize-cryptic'
    },
    'toughie': {
        'name': 'Toughie',
        'api_path': 'toughie-crossword/toughie-crossword',
        'offset': 77743,
        'reference_date': date(2026, 1, 23),  # Thursday
        'reference_number': 3624,
        'days': [1, 2, 3, 4],  # Tue-Fri
        'source': 'telegraph-toughie'
    },
    'prize-toughie': {
        'name': 'Prize Toughie',
        'api_path': 'prize-toughie/prize-toughie',
        'offset': 81447,
        'reference_date': date(2026, 1, 18),  # Sunday
        'reference_number': 208,
        'days': [6],  # Sunday only
        'source': 'telegraph-prize-toughie'
    }
}


def to_iso_date(date_str):
    """Convert 'Friday, 13 February 2026' to '2026-02-13'. Pass through if already ISO."""
    if not date_str or (len(date_str) == 10 and date_str[4] == '-'):
        return date_str
    try:
        return datetime.strptime(date_str, '%A, %d %B %Y').strftime('%Y-%m-%d')
    except ValueError:
        return date_str


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

    # Count matching days between reference and target
    diff = count_matching_days(ref_date, target_date, weekdays)
    return ref_number + diff


def puzzle_number_to_api_id(puzzle_type, puzzle_number):
    """Convert puzzle number to API ID."""
    return puzzle_number + PUZZLE_TYPES[puzzle_type]['offset']


def get_puzzle_data(puzzle_type, api_id):
    """Fetch puzzle data from Telegraph API."""
    api_path = PUZZLE_TYPES[puzzle_type]['api_path']
    url = f"https://puzzlesdata.telegraph.co.uk/puzzles/{api_path}-{api_id}.json"
    print(f"Fetching: {url}")

    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error: HTTP {response.status_code}")
        return None

    return response.json()


def parse_puzzle(data, puzzle_type):
    """Parse the API response."""
    puzzle_json = data.get('json', {})
    copy = puzzle_json.get('copy', {})

    title = copy.get('title', '')
    date_publish = to_iso_date(copy.get('date-publish', ''))

    # Extract puzzle number from title (e.g., "Cryptic Crossword No 31144")
    match = re.search(r'No\s*(\d+)', title)
    puzzle_number = int(match.group(1)) if match else copy.get('id', 0)

    # Clues are in copy.clues as a list: [Across group, Down group]
    clues_groups = copy.get('clues', [])

    across = []
    down = []

    for group in clues_groups:
        direction = group.get('title', '').lower()  # 'Across' or 'Down'
        clue_list = group.get('clues', [])

        for clue in clue_list:
            clue_obj = {
                'number': clue.get('number', ''),
                'clue': unescape(clue.get('clue', '')),
                'answer': clue.get('answer', ''),
                'enumeration': clue.get('format', '')
            }

            if direction == 'across':
                across.append(clue_obj)
            elif direction == 'down':
                down.append(clue_obj)

    return {
        'puzzle_type': puzzle_type,
        'puzzle_number': puzzle_number,
        'title': title,
        'date': date_publish,
        'across': across,
        'down': down
    }


def save_to_database(puzzle_data, puzzle_type):
    """Save puzzle clues to telegraph_clues table."""
    print(f"Saving to telegraph_clues table...")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS telegraph_clues (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            puzzle_type TEXT,
            puzzle_number TEXT,
            puzzle_date TEXT,
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
        CREATE INDEX IF NOT EXISTS idx_telegraph_puzzle 
        ON telegraph_clues(puzzle_type, puzzle_number)
    """)

    puzzle_number = puzzle_data.get('puzzle_number', 0)
    puzzle_date = puzzle_data.get('date', '')
    fetched_at = datetime.now().isoformat()

    # Check if already fetched
    cursor.execute("""
        SELECT COUNT(*) FROM telegraph_clues 
        WHERE puzzle_type = ? AND puzzle_number = ?
    """, (puzzle_type, str(puzzle_number)))

    if cursor.fetchone()[0] > 0:
        print(f"Puzzle {puzzle_number} already in database - skipping")
        conn.close()
        return 0

    clue_count = 0
    for direction in ['across', 'down']:
        for clue in puzzle_data.get(direction, []):
            cursor.execute("""
                INSERT INTO telegraph_clues 
                (puzzle_type, puzzle_number, puzzle_date, clue_number, direction, 
                 clue_text, enumeration, answer, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                puzzle_type,
                str(puzzle_number),
                puzzle_date,
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


def publish_to_main(puzzle_type, puzzle_number):
    """Copy completed puzzle from telegraph_clues to main clues table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    source = PUZZLE_TYPES[puzzle_type]['source']

    # Get clues that have explanations and aren't published yet
    cursor.execute("""
        SELECT puzzle_number, clue_number, direction, clue_text, enumeration, answer
        FROM telegraph_clues
        WHERE puzzle_type = ? AND puzzle_number = ? 
        AND explanation IS NOT NULL AND published = 0
    """, (puzzle_type, str(puzzle_number)))

    clues = cursor.fetchall()
    if not clues:
        print(f"No unpublished clues with explanations for {puzzle_type} {puzzle_number}")
        conn.close()
        return 0

    # Insert into main clues table
    for clue in clues:
        cursor.execute("""
            INSERT INTO clues 
            (puzzle_number, clue_number, direction, clue_text, enumeration, answer, source)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (*clue, source))

    # Mark as published
    cursor.execute("""
        UPDATE telegraph_clues SET published = 1
        WHERE puzzle_type = ? AND puzzle_number = ?
    """, (puzzle_type, str(puzzle_number)))

    conn.commit()
    conn.close()

    print(f"Published {len(clues)} clues to main table")
    return len(clues)


def get_todays_puzzles():
    """Get list of puzzle types available today."""
    today = date.today()
    weekday = today.weekday()

    available = []
    for puzzle_type, config in PUZZLE_TYPES.items():
        if weekday in config['days']:
            available.append(puzzle_type)

    return available


def puzzle_already_fetched(puzzle_type, puzzle_number):
    """Check if puzzle is already in database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Ensure table exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS telegraph_clues (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            puzzle_type TEXT,
            puzzle_number TEXT,
            puzzle_date TEXT,
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
        SELECT COUNT(*) FROM telegraph_clues 
        WHERE puzzle_type = ? AND puzzle_number = ?
    """, (puzzle_type, str(puzzle_number)))

    count = cursor.fetchone()[0]
    conn.close()
    return count > 0


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

    api_id = puzzle_number_to_api_id(puzzle_type, puzzle_number)

    print(f"\n{'=' * 50}")
    print(f"{config['name']}")
    print(f"Puzzle #: {puzzle_number} | API ID: {api_id}")

    data = get_puzzle_data(puzzle_type, api_id)
    if not data:
        return None

    puzzle = parse_puzzle(data, puzzle_type)

    print(f"Title: {puzzle.get('title')}")
    print(f"Date: {puzzle.get('date')}")
    print(
        f"Clues: {len(puzzle.get('across', []))} across, {len(puzzle.get('down', []))} down")

    if puzzle.get('across'):
        first = puzzle['across'][0]
        print(f"Sample: {first['number']}. {first['clue'][:40]}... = {first['answer']}")

    save_to_database(puzzle, puzzle_type)

    json_path = f"telegraph_{puzzle_type}_{puzzle.get('puzzle_number')}.json"
    with open(json_path, 'w') as f:
        json.dump(puzzle, f, indent=2)
    print(f"JSON: {json_path}")

    return puzzle


def main():
    print("=" * 60)
    print("TELEGRAPH PUZZLE SCRAPER")
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
            # Show status of puzzles in database
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT puzzle_type, puzzle_number, puzzle_date,
                       COUNT(*) as clues,
                       SUM(CASE WHEN explanation IS NOT NULL THEN 1 ELSE 0 END) as explained,
                       MAX(published) as published
                FROM telegraph_clues
                GROUP BY puzzle_type, puzzle_number
                ORDER BY fetched_at DESC
                LIMIT 20
            """)
            rows = cursor.fetchall()
            conn.close()

            print("\nRecent puzzles in telegraph_clues:")
            print(
                f"{'Type':15} {'#':>8} {'Date':>12} {'Clues':>6} {'Expl':>6} {'Pub':>4}")
            print("-" * 55)
            for row in rows:
                pub = "Yes" if row[5] else "No"
                print(
                    f"{row[0]:15} {row[1]:>8} {row[2]:>12} {row[3]:>6} {row[4]:>6} {pub:>4}")

        elif arg == '--publish':
            # Publish a puzzle to main table
            if len(sys.argv) < 4:
                print("Usage: --publish <puzzle_type> <puzzle_number>")
            else:
                publish_to_main(sys.argv[2], sys.argv[3])

        elif arg in PUZZLE_TYPES:
            puzzle_number = int(sys.argv[2]) if len(sys.argv) > 2 else None
            fetch_puzzle(arg, puzzle_number)

        else:
            try:
                puzzle_number = int(arg)
                fetch_puzzle('cryptic', puzzle_number)
            except ValueError:
                print(f"\nUsage:")
                print("  python telegraph_all.py              # Today's daily cryptic")
                print(
                    "  python telegraph_all.py --all        # All puzzles available today")
                print("  python telegraph_all.py --list       # List puzzle types")
                print("  python telegraph_all.py --status     # Show puzzles in database")
                print("  python telegraph_all.py cryptic      # Today's daily cryptic")
                print("  python telegraph_all.py cryptic 31144  # Specific puzzle number")
                print("  python telegraph_all.py toughie      # Today's toughie")
                print("  python telegraph_all.py prize-cryptic  # Latest prize cryptic")
    else:
        # Default: fetch all puzzles available today
        for pt in available:
            fetch_puzzle(pt)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()