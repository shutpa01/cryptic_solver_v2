#!/usr/bin/env python3
"""Guardian Puzzle Scraper - All Puzzle Types
Fetches puzzles from Guardian JSON API and saves directly to clues table in clues_master.db.
Also fetches Observer Everyman from the slowdownwiseup API.
"""

import requests
import json
import sqlite3
import os
import sys
import re
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from dotenv import load_dotenv
from html import unescape

load_dotenv()

DB_PATH = os.getenv('DB_PATH',
                    r"C:\Users\shute\PycharmProjects\cryptic_solver_V2\data\clues_master.db")

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
    'everyman': {
        'name': 'Observer Everyman',
        'url_path': 'everyman',
        'reference_date': date(2026, 4, 12),  # Sunday
        'reference_number': 4147,
        'days': [6],  # Sunday only
        'source': 'guardian-everyman',
        'api': 'observer',  # flag: use Observer API, not Guardian
    },
}

# Observer Everyman API
OBSERVER_ARTICLE_URL = "https://observer.co.uk/puzzles/everyman/article/everyman-no-{}"
OBSERVER_DATA_URL = "https://content-api.slowdownwiseup.co.uk/api/mobile/v1/puzzle-data/{}/file/data.json"


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

    for attempt in range(3):
        try:
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                print(f"Error: HTTP {response.status_code}")
                return None
            return response.json()
        except Exception as e:
            if attempt < 2:
                wait = 2 * (attempt + 1)
                print(f"  Request failed (attempt {attempt + 1}/3), retrying in {wait}s: {e}")
                time.sleep(wait)
            else:
                print(f"  Request failed after 3 attempts: {e}")
                return None


def build_grid_solution(data):
    """Build a flat solution string from the Guardian API crossword data.

    Returns (solution_string, rows, cols) or (None, 15, 15) if not enough data.
    """
    crossword = data.get('crossword', {})
    dims = crossword.get('dimensions', {})
    rows = dims.get('rows', 15)
    cols = dims.get('cols', 15)
    entries = crossword.get('entries', [])

    if not entries:
        return None, rows, cols

    grid = [[' '] * cols for _ in range(rows)]
    has_any = False

    for entry in entries:
        sol = entry.get('solution', '')
        if not sol:
            continue
        pos = entry.get('position', {})
        x, y = pos.get('x', 0), pos.get('y', 0)
        direction = entry.get('direction', '')

        for i, ch in enumerate(sol):
            if direction == 'across':
                c = x + i
                if c < cols:
                    grid[y][c] = ch
                    has_any = True
            else:
                r = y + i
                if r < rows:
                    grid[r][x] = ch
                    has_any = True

    if not has_any:
        return None, rows, cols

    solution = ''.join(''.join(row) for row in grid)
    return solution, rows, cols


def parse_puzzle(data, puzzle_type):
    """Parse the Guardian API response."""
    crossword = data.get('crossword', {})

    puzzle_number = crossword.get('number', 0)
    name = crossword.get('name', '')
    setter = crossword.get('creator', {}).get('name', '')
    puzzle_date = crossword.get('date', '')

    # Convert epoch to ISO date
    if isinstance(puzzle_date, int):
        puzzle_date = datetime.utcfromtimestamp(puzzle_date / 1000).strftime('%Y-%m-%d')
    elif isinstance(puzzle_date, str) and ',' in puzzle_date:
        # Convert "Friday, 13 February 2026" to "2026-02-13"
        try:
            puzzle_date = datetime.strptime(puzzle_date, '%A, %d %B %Y').strftime('%Y-%m-%d')
        except ValueError:
            pass

    entries = crossword.get('entries', [])

    # Build group map for spanning clues — combine solutions
    group_solutions = {}  # group_leader_id -> combined solution
    for entry in entries:
        group = entry.get('group', [])
        if len(group) > 1:
            leader = group[0]
            if leader not in group_solutions:
                group_solutions[leader] = {}
            entry_id = entry.get('id', '')
            group_solutions[leader][entry_id] = {
                'solution': entry.get('solution', ''),
                'order': group.index(entry_id) if entry_id in group else 0,
            }

    across = []
    down = []

    for entry in entries:
        clue_text = entry.get('clue', '')
        # Remove HTML tags and decode entities
        clue_text = re.sub(r'<[^>]+>', '', clue_text)
        clue_text = unescape(clue_text)

        # Extract enumeration from end of clue
        enum_match = re.search(r'\(([0-9,\-\s]+)\)\s*$', clue_text)
        enumeration = enum_match.group(1) if enum_match else str(entry.get('length', ''))

        # Remove enumeration from clue text
        clean_text = re.sub(r'\s*\([0-9,\-\s]+\)\s*$', '', clue_text).strip()

        # For spanning clues, combine the full answer on the leader entry
        entry_id = entry.get('id', '')
        group = entry.get('group', [])
        answer = entry.get('solution', '')

        if len(group) > 1 and group[0] == entry_id:
            # This is the leader — combine all group solutions in order
            group_data = group_solutions.get(entry_id, {})
            parts = sorted(group_data.values(), key=lambda x: x['order'])
            answer = ''.join(p['solution'] for p in parts)

        clue_obj = {
            'number': entry.get('number', ''),
            'clue': clean_text,
            'answer': answer,
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
    """Check if puzzle is already in the clues table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*) FROM clues
        WHERE source = 'guardian' AND puzzle_number = ?
    """, (str(puzzle_number),))

    count = cursor.fetchone()[0]
    conn.close()
    return count > 0


# ── Observer Everyman functions ──────────────────────────────────────────


def fetch_everyman_uuid(puzzle_number):
    """Scrape the Observer article page to find the puzzle data UUID."""
    url = OBSERVER_ARTICLE_URL.format(puzzle_number)
    print(f"  Fetching article: {url}")
    try:
        r = requests.get(url, timeout=15, headers={'User-Agent': 'Mozilla/5.0'})
        if r.status_code != 200:
            print(f"  Article HTTP {r.status_code}")
            return None
        uuids = set(re.findall(
            r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
            r.text))
        # Test each UUID against the puzzle-data API
        for uuid in uuids:
            dr = requests.get(OBSERVER_DATA_URL.format(uuid), timeout=10)
            if dr.status_code == 200:
                data = dr.json()
                title = data.get('headline', '')
                if 'veryman' in title.lower() or 'Everyman' in title:
                    return uuid
        print(f"  No puzzle UUID found (tried {len(uuids)})")
        return None
    except Exception as e:
        print(f"  Error fetching article: {e}")
        return None


def fetch_everyman_data(uuid):
    """Fetch puzzle JSON from the Observer/slowdownwiseup API."""
    url = OBSERVER_DATA_URL.format(uuid)
    try:
        r = requests.get(url, timeout=15)
        if r.status_code != 200:
            print(f"  Data API HTTP {r.status_code}")
            return None
        return r.json()
    except Exception as e:
        print(f"  Error fetching data: {e}")
        return None


def parse_everyman_puzzle(data):
    """Parse Observer Everyman JSON into our standard puzzle structure."""
    copy = data.get('copy', {})
    title = copy.get('title', '')

    # Extract puzzle number from title like "Everyman No. 4147"
    m = re.search(r'(\d{4,})', title)
    puzzle_number = int(m.group(1)) if m else 0

    # Parse date
    date_str = copy.get('date-publish', '')
    puzzle_date = ''
    if date_str:
        try:
            puzzle_date = datetime.strptime(
                date_str, '%A, %d %B %Y').strftime('%Y-%m-%d')
        except ValueError:
            pass

    # Parse clues
    clue_sections = copy.get('clues', [])
    across = []
    down = []
    for section in clue_sections:
        direction = 'across' if section.get('title', '').lower() == 'across' else 'down'
        target = across if direction == 'across' else down
        for clue in section.get('clues', []):
            target.append({
                'number': clue.get('number', 0),
                'clue': unescape(clue.get('clue', '')),
                'answer': clue.get('answer', ''),
                'enumeration': clue.get('format', ''),
            })

    return {
        'puzzle_number': puzzle_number,
        'title': title,
        'setter': copy.get('setter', ''),
        'date': puzzle_date,
        'across': across,
        'down': down,
    }


def build_everyman_grid(data):
    """Extract grid solution from Observer JSON.
    If no solution yet (current week's puzzle), builds a structure grid
    from word positions — white cells as dots, black cells as spaces.
    Returns (solution_string, rows, cols) or (None, 15, 15)."""
    copy = data.get('copy', {})
    settings = copy.get('settings', {})
    solution = settings.get('solution', '')
    gridsize = copy.get('gridsize', {})
    rows = int(gridsize.get('rows', 15))
    cols = int(gridsize.get('cols', 15))

    if solution and len(solution) == rows * cols:
        return solution, rows, cols

    # No solution yet — build structure from word positions
    words = copy.get('words', [])
    if not words:
        return None, rows, cols

    grid = [[' '] * cols for _ in range(rows)]  # space = black cell
    for word in words:
        x_spec = str(word.get('x', ''))
        y_spec = str(word.get('y', ''))
        if '-' in x_spec:
            # Across: x is range, y is row
            x_parts = x_spec.split('-')
            x_start, x_end = int(x_parts[0]) - 1, int(x_parts[1]) - 1
            row = int(y_spec) - 1
            for c in range(x_start, x_end + 1):
                if 0 <= row < rows and 0 <= c < cols:
                    grid[row][c] = '.'
        elif '-' in y_spec:
            # Down: y is range, x is column
            y_parts = y_spec.split('-')
            y_start, y_end = int(y_parts[0]) - 1, int(y_parts[1]) - 1
            col = int(x_spec) - 1
            for r in range(y_start, y_end + 1):
                if 0 <= r < rows and 0 <= col < cols:
                    grid[r][col] = '.'

    structure = ''.join(''.join(row) for row in grid)
    return structure, rows, cols


def parse_previous_solution(prev_solution_str, rows=15, cols=15):
    """Parse the previous_solution field into a flat grid string.
    The field is formatted with spaces for black cells and newlines between rows."""
    if not prev_solution_str:
        return None
    # The previous_solution is already a grid laid out as rows×cols characters
    # with spaces for black cells, but may have newlines
    flat = prev_solution_str.replace('\n', '').replace('\r', '')
    if len(flat) == rows * cols:
        return flat
    return None


def extract_answers_from_grid(grid_str, words, rows=15, cols=15):
    """Extract answers from a flat grid string using word positions.
    words: list of {id, x, y} from the Observer JSON."""
    if not grid_str or len(grid_str) != rows * cols:
        return {}

    def get_cell(r, c):
        if 0 <= r < rows and 0 <= c < cols:
            ch = grid_str[r * cols + c]
            return ch if ch != ' ' else None
        return None

    answers = {}  # word_id -> answer string
    for word in words:
        wid = word.get('id')
        x_spec = str(word.get('x', ''))
        y_spec = str(word.get('y', ''))

        if '-' in x_spec:
            # Across word: x is range like "4-7", y is row
            x_parts = x_spec.split('-')
            x_start, x_end = int(x_parts[0]) - 1, int(x_parts[1]) - 1
            row = int(y_spec) - 1
            ans = ''
            for c in range(x_start, x_end + 1):
                ch = get_cell(row, c)
                if ch:
                    ans += ch
            answers[wid] = ans
        elif '-' in y_spec:
            # Down word: y is range like "1-7", x is column
            y_parts = y_spec.split('-')
            y_start, y_end = int(y_parts[0]) - 1, int(y_parts[1]) - 1
            col = int(x_spec) - 1
            ans = ''
            for r in range(y_start, y_end + 1):
                ch = get_cell(r, col)
                if ch:
                    ans += ch
            answers[wid] = ans

    return answers


def backfill_everyman_previous(data, current_number):
    """Use this week's previous_solution to backfill last week's answers and grid."""
    copy = data.get('copy', {})
    settings = copy.get('settings', {})
    prev_solution_str = settings.get('previous_solution', '')
    prev_title = settings.get('previous_title', '')

    if not prev_solution_str:
        print("  No previous_solution in JSON")
        return

    prev_number = current_number - 1
    # Confirm from title if possible
    m = re.search(r'(\d{4,})', prev_title.replace(',', ''))
    if m:
        prev_number = int(m.group(1))

    gridsize = copy.get('gridsize', {})
    rows = int(gridsize.get('rows', 15))
    cols = int(gridsize.get('cols', 15))

    grid_str = parse_previous_solution(prev_solution_str, rows, cols)
    if not grid_str:
        print(f"  Could not parse previous_solution (len={len(prev_solution_str)})")
        return

    print(f"  Backfilling #{prev_number} from previous_solution")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Save grid solution for previous puzzle (replaces structure-only grid)
    cursor.execute("""
        INSERT INTO puzzle_grids
        (source, puzzle_number, solution, grid_rows, grid_cols, api_folder)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(source, puzzle_number) DO UPDATE SET
            solution = excluded.solution
    """, ('guardian', str(prev_number), grid_str, rows, cols,
          f'observer-everyman-{prev_number}'))

    # Extract answers from the grid using word positions
    # We need the PREVIOUS puzzle's word layout, but we don't have it.
    # Instead, match answers to clues by reading the grid at clue positions.
    # The clues for the previous puzzle should already be in the DB.
    # We can reconstruct answers from the grid + clue positions.

    # Get existing clues for previous puzzle
    existing = cursor.execute("""
        SELECT clue_number, direction, answer
        FROM clues
        WHERE source = 'guardian' AND puzzle_number = ?
    """, (str(prev_number),)).fetchall()

    if not existing:
        print(f"  No clues found for #{prev_number} in DB — skipping answer backfill")
        conn.commit()
        conn.close()
        return

    # Build answer map from the grid by reading word positions
    # We need the previous puzzle's word layout. Fetch its article to get it.
    prev_uuid = fetch_everyman_uuid(prev_number)
    if prev_uuid:
        prev_data = fetch_everyman_data(prev_uuid)
        if prev_data:
            prev_copy = prev_data.get('copy', {})
            prev_words = prev_copy.get('words', [])
            prev_clue_sections = prev_copy.get('clues', [])

            word_answers = extract_answers_from_grid(grid_str, prev_words, rows, cols)

            # Map word_id to (clue_number, direction)
            for section in prev_clue_sections:
                direction = 'across' if section.get('title', '').lower() == 'across' else 'down'
                for clue in section.get('clues', []):
                    wid = clue.get('word')
                    cnum = str(clue.get('number', ''))
                    answer = word_answers.get(wid, '')
                    if answer:
                        cursor.execute("""
                            UPDATE clues SET answer = ?
                            WHERE source = 'guardian' AND puzzle_number = ?
                              AND clue_number = ? AND direction = ?
                              AND (answer IS NULL OR answer = '')
                        """, (answer, str(prev_number), cnum, direction))

            updated = cursor.rowcount
            print(f"  Backfilled answers for #{prev_number}")

    conn.commit()
    conn.close()


def fetch_everyman_new():
    """Fetch new Observer Everyman puzzles and backfill previous answers."""
    config = PUZZLE_TYPES['everyman']
    last_number = get_last_puzzle_number('everyman')

    if last_number is None:
        last_number = config['reference_number'] - 1
        print(f"\n{config['name']}: No puzzles in DB — starting from #{last_number + 1}")
    else:
        print(f"\n{config['name']}: last in DB is #{last_number}")

    fetched = []
    consecutive_misses = 0
    number = last_number + 1

    while consecutive_misses < 2:
        if puzzle_already_fetched('everyman', number):
            # Even if puzzle exists, check if we need to backfill previous answers
            number += 1
            continue

        print(f"  Trying #{number}...")
        uuid = fetch_everyman_uuid(number)
        if not uuid:
            consecutive_misses += 1
            number += 1
            continue

        data = fetch_everyman_data(uuid)
        if not data:
            consecutive_misses += 1
            number += 1
            continue

        puzzle = parse_everyman_puzzle(data)
        parsed_number = puzzle.get('puzzle_number')
        if not parsed_number:
            consecutive_misses += 1
            number += 1
            continue

        # The article URL for an unpublished number can return 200 with related-
        # article UUIDs in the HTML; fetch_everyman_uuid then picks one of those
        # and we end up with data for an OLDER puzzle. Treat that as a miss
        # instead of silently overwriting the older puzzle's row.
        if parsed_number != number:
            print(f"  Article #{number} returned data for #{parsed_number} - "
                  f"requested puzzle not yet published, skipping")
            consecutive_misses += 1
            number += 1
            continue

        consecutive_misses = 0

        # Save this week's clues (answers may be empty)
        source_url = OBSERVER_DATA_URL.format(uuid)
        save_to_database(puzzle, 'everyman', source_url=source_url)

        # Save Observer JSON for grid rendering (build_grid_from_json needs it)
        json_dir = Path(__file__).resolve().parent
        json_path = json_dir / f"guardian_everyman_{puzzle['puzzle_number']}.json"
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

        # Save grid if solution is available
        solution, grid_rows, grid_cols = build_everyman_grid(data)
        if solution:
            conn = sqlite3.connect(DB_PATH)
            conn.execute("""
                INSERT INTO puzzle_grids
                (source, puzzle_number, solution, grid_rows, grid_cols, api_folder)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(source, puzzle_number) DO UPDATE SET
                    solution = COALESCE(excluded.solution, puzzle_grids.solution)
            """, ('guardian', str(puzzle['puzzle_number']), solution,
                  grid_rows, grid_cols, source_url))
            conn.commit()
            conn.close()
            print(f"  Grid saved for #{puzzle['puzzle_number']}")

        clue_count = len(puzzle.get('across', [])) + len(puzzle.get('down', []))
        print(f"  #{number}: {clue_count} clues ({puzzle.get('date', '?')})")

        # Backfill previous week's answers
        backfill_everyman_previous(data, number)

        fetched.append(puzzle)
        number += 1

    if fetched:
        print(f"  -> {len(fetched)} new Everyman puzzle(s)")
    else:
        print(f"  -> up to date")

    return fetched


def save_to_database(puzzle_data, puzzle_type, raw_api_data=None, source_url=None):
    """Save puzzle clues to clues table and grid to puzzle_grids."""
    print(f"Saving to clues table (source='guardian')...")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    puzzle_number = puzzle_data.get('puzzle_number', 0)
    puzzle_date = puzzle_data.get('date', '')

    clue_count = 0
    for direction in ['across', 'down']:
        for clue in puzzle_data.get(direction, []):
            answer = clue.get('answer', '')
            cursor.execute("""
                INSERT INTO clues
                (source, puzzle_number, publication_date, clue_number, direction,
                 clue_text, enumeration, answer)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source, puzzle_number, clue_number, direction, publication_date)
                DO UPDATE SET answer = CASE
                    WHEN length(excluded.answer) > length(COALESCE(clues.answer, ''))
                    THEN excluded.answer
                    WHEN clues.answer IS NULL OR clues.answer = ''
                    THEN excluded.answer
                    ELSE clues.answer
                END
            """, (
                'guardian',
                str(puzzle_number),
                puzzle_date,
                str(clue.get('number', '')),
                direction,
                clue.get('clue', ''),
                clue.get('enumeration', ''),
                answer,
            ))
            clue_count += 1

    # Save grid solution if we have raw API data
    if raw_api_data:
        solution, grid_rows, grid_cols = build_grid_solution(raw_api_data)
        if solution and len(solution) == grid_rows * grid_cols:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS puzzle_grids (
                    source TEXT NOT NULL,
                    puzzle_number TEXT NOT NULL,
                    solution TEXT,
                    grid_rows INTEGER NOT NULL DEFAULT 15,
                    grid_cols INTEGER NOT NULL DEFAULT 15,
                    api_folder TEXT,
                    api_type TEXT,
                    api_id TEXT,
                    PRIMARY KEY (source, puzzle_number)
                )
            """)
            # api_folder stores the source URL
            cursor.execute("""
                INSERT INTO puzzle_grids
                (source, puzzle_number, solution, grid_rows, grid_cols, api_folder)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(source, puzzle_number) DO UPDATE SET
                    solution = COALESCE(excluded.solution, puzzle_grids.solution),
                    api_folder = COALESCE(excluded.api_folder, puzzle_grids.api_folder)
            """, (
                'guardian', str(puzzle_number),
                solution, grid_rows, grid_cols, source_url,
            ))
            print(f"Saved grid solution ({grid_rows}x{grid_cols})")

    conn.commit()
    conn.close()

    print(f"Saved {clue_count} clues")
    return clue_count


def get_last_puzzle_number(puzzle_type):
    """Get the highest puzzle number in the clues table for this puzzle type's series."""
    config = PUZZLE_TYPES[puzzle_type]
    ref = config['reference_number']
    # Infer number range: reference_number ± generous margin
    lo = max(1, ref - 500)
    hi = ref + 500

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT MAX(CAST(puzzle_number AS INTEGER)) FROM clues
        WHERE source = 'guardian'
          AND CAST(puzzle_number AS INTEGER) BETWEEN ? AND ?
    """, (lo, hi))
    result = cursor.fetchone()[0]
    conn.close()
    return result


def fetch_new_puzzles(puzzle_type):
    """Fetch all new puzzles by incrementing from the last known number.
    Keeps going until 3 consecutive 404s to catch up on any missed days."""
    # Observer Everyman uses a completely different API
    if PUZZLE_TYPES[puzzle_type].get('api') == 'observer':
        return fetch_everyman_new()

    config = PUZZLE_TYPES[puzzle_type]
    last_number = get_last_puzzle_number(puzzle_type)

    if last_number is None:
        print(f"\n{config['name']}: No puzzles in DB — use manual fetch to seed")
        return []

    print(f"\n{config['name']}: last in DB is #{last_number}")

    fetched = []
    consecutive_misses = 0
    number = last_number + 1

    while consecutive_misses < 3:
        if puzzle_already_fetched(puzzle_type, number):
            number += 1
            continue

        data = get_puzzle_data(puzzle_type, number)
        if data:
            puzzle = parse_puzzle(data, puzzle_type)
            url_path = PUZZLE_TYPES[puzzle_type]['url_path']
            api_url = f"https://www.theguardian.com/crosswords/{url_path}/{number}.json"
            save_to_database(puzzle, puzzle_type, raw_api_data=data, source_url=api_url)

            json_path = f"guardian_{puzzle_type}_{puzzle.get('puzzle_number')}.json"
            with open(json_path, 'w') as f:
                json.dump(puzzle, f, indent=2)

            clue_count = len(puzzle.get('across', [])) + len(puzzle.get('down', []))
            print(f"  #{number}: {clue_count} clues ({puzzle.get('date', '?')})")
            fetched.append(puzzle)
            consecutive_misses = 0
        else:
            consecutive_misses += 1

        number += 1

    if fetched:
        print(f"  -> {len(fetched)} new puzzle(s)")
    else:
        print(f"  -> up to date")

    return fetched


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

    # Observer Everyman uses a different API
    if config.get('api') == 'observer':
        return fetch_everyman_new()

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

    url_path = PUZZLE_TYPES[puzzle_type]['url_path']
    api_url = f"https://www.theguardian.com/crosswords/{url_path}/{puzzle_number}.json"
    save_to_database(puzzle, puzzle_type, raw_api_data=data, source_url=api_url)

    json_path = f"guardian_{puzzle_type}_{puzzle.get('puzzle_number')}.json"
    with open(json_path, 'w') as f:
        json.dump(puzzle, f, indent=2)
    print(f"JSON: {json_path}")

    return puzzle


def backfill_grids():
    """Backfill grid solutions and spanning answers for all Guardian puzzles in DB."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Ensure puzzle_grids table exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS puzzle_grids (
            source TEXT NOT NULL,
            puzzle_number TEXT NOT NULL,
            solution TEXT,
            grid_rows INTEGER NOT NULL DEFAULT 15,
            grid_cols INTEGER NOT NULL DEFAULT 15,
            api_folder TEXT,
            api_type TEXT,
            api_id TEXT,
            PRIMARY KEY (source, puzzle_number)
        )
    """)
    conn.commit()

    # Find Guardian puzzles without grids
    cursor.execute("""
        SELECT DISTINCT c.puzzle_number
        FROM clues c
        LEFT JOIN puzzle_grids g ON g.source = 'guardian' AND g.puzzle_number = c.puzzle_number
        WHERE c.source = 'guardian' AND g.puzzle_number IS NULL
        ORDER BY CAST(c.puzzle_number AS INTEGER)
    """)
    missing = [row[0] for row in cursor.fetchall()]
    conn.close()

    print(f"Guardian puzzles without grids: {len(missing)}")
    if not missing:
        return

    filled = 0
    no_solution = 0
    errors = 0

    for puzzle_number in missing:
        # Try to fetch from Guardian API
        # Guess puzzle type from number range
        pnum = int(puzzle_number)
        puzzle_type = None
        for pt, config in PUZZLE_TYPES.items():
            ref = config['reference_number']
            if abs(pnum - ref) < 500:
                puzzle_type = pt
                break

        if puzzle_type is None:
            # Try all types
            for pt in ['cryptic', 'prize', 'quick-cryptic', 'everyman', 'quiptic']:
                data = get_puzzle_data(pt, pnum)
                if data:
                    puzzle_type = pt
                    break
            else:
                errors += 1
                continue
        else:
            data = get_puzzle_data(puzzle_type, pnum)

        if not data:
            errors += 1
            continue

        solution, grid_rows, grid_cols = build_grid_solution(data)
        if not solution or len(solution) != grid_rows * grid_cols:
            no_solution += 1
            print(f"  #{puzzle_number}: no solution available (prize?)")
            continue

        # Save grid
        conn2 = sqlite3.connect(DB_PATH)
        c2 = conn2.cursor()
        c2.execute("""
            INSERT INTO puzzle_grids (source, puzzle_number, solution, grid_rows, grid_cols)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(source, puzzle_number) DO UPDATE SET
                solution = COALESCE(excluded.solution, puzzle_grids.solution)
        """, ('guardian', str(puzzle_number), solution, grid_rows, grid_cols))

        # Also fix spanning answers
        puzzle = parse_puzzle(data, puzzle_type)
        for direction in ['across', 'down']:
            for clue in puzzle.get(direction, []):
                answer = clue.get('answer', '')
                c2.execute("""
                    UPDATE clues SET answer = ?
                    WHERE source = 'guardian' AND puzzle_number = ?
                      AND clue_number = ? AND direction = ?
                      AND length(?) > length(COALESCE(answer, ''))
                """, (answer, str(puzzle_number),
                      str(clue.get('number', '')), direction, answer))

        conn2.commit()
        conn2.close()
        filled += 1
        print(f"  #{puzzle_number}: grid saved")

        time.sleep(0.5)  # Be polite to Guardian API

    print(f"\nBackfill complete: {filled} grids added, {no_solution} without solutions, {errors} errors")


def main():
    print("=" * 60)
    print("GUARDIAN PUZZLE SCRAPER")
    print("=" * 60)

    today = date.today()
    print(f"Today: {today.strftime('%A, %d %B %Y')}")
    print(f"Database: {DB_PATH}")

    available = get_todays_puzzles()
    print(f"Available today: {', '.join(available) if available else 'None'}")

    if len(sys.argv) > 1:
        arg = sys.argv[1]

        if arg == '--all':
            for pt in PUZZLE_TYPES:
                fetch_new_puzzles(pt)

        elif arg == '--backfill-grids':
            backfill_grids()

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
                SELECT puzzle_number, publication_date,
                       COUNT(*) as clues,
                       SUM(CASE WHEN explanation IS NOT NULL AND explanation != '' THEN 1 ELSE 0 END) as explained
                FROM clues
                WHERE source = 'guardian'
                GROUP BY puzzle_number
                ORDER BY CAST(puzzle_number AS INTEGER) DESC
                LIMIT 20
            """)
            rows = cursor.fetchall()
            conn.close()

            print("\nRecent guardian puzzles in clues table:")
            print(f"{'#':>8} {'Date':12} {'Clues':>6} {'Expl':>6}")
            print("-" * 35)
            for row in rows:
                print(f"{row[0]:>8} {(row[1] or ''):12} {row[2]:>6} {row[3]:>6}")

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
        # Default: catch up on all puzzle types from last known number
        for pt in PUZZLE_TYPES:
            fetch_new_puzzles(pt)

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
