#!/usr/bin/env python3
"""Independent Edition Crossword Scraper

Scrapes cryptic crosswords from edition.independent.co.uk (Pugpig platform).
This source provides structured JSON with proper clue numbers, directions,
and grid positions — unlike the old independentcrossword.co.uk scraper.

Data flow:
    date → edition ID (DDMMYY) → content.xml → find cryptic crossword entry
    → fetch puzzle HTML → extract inline JSON → compute clue numbers → save

Usage:
    python independent_edition.py                # Today's puzzle
    python independent_edition.py --date 2025-10-06  # Specific date
    python independent_edition.py --dry-run      # Preview without DB writes
"""

import json
import os
import re
import sqlite3
import sys
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv('DB_PATH',
                    r"C:\Users\shute\PycharmProjects\cryptic_solver_V2\data\clues_master.db")

EDITION_BASE = "https://edition.independent.co.uk"
ATOM_NS = "http://www.w3.org/2005/Atom"

REQUEST_DELAY = 1  # seconds between requests


def edition_id_from_date(puzzle_date):
    """Construct the Pugpig edition ID from a date.

    Format: uk.co.independent.issue.DDMMYY
    """
    return f"uk.co.independent.issue.{puzzle_date.strftime('%d%m%y')}"


def content_xml_url(puzzle_date):
    """Construct the content.xml URL for a given date."""
    eid = edition_id_from_date(puzzle_date)
    return f"{EDITION_BASE}/edition/{eid}/content.xml"


def fetch_content_xml(puzzle_date):
    """Fetch and parse the edition's content.xml.

    Returns the parsed XML root or None on failure.
    """
    url = content_xml_url(puzzle_date)
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            return ET.fromstring(resp.content)
        else:
            print(f"  content.xml HTTP {resp.status_code}: {url}")
            return None
    except Exception as e:
        print(f"  Error fetching content.xml: {e}")
        return None


def find_cryptic_puzzle_url(xml_root):
    """Find the cryptic crossword puzzle URL from content.xml.

    Returns (puzzle_url, puzzle_title) or (None, None) if not found.
    """
    for entry in xml_root.findall(f'{{{ATOM_NS}}}entry'):
        title_el = entry.find(f'{{{ATOM_NS}}}title')
        if title_el is None:
            continue
        title = title_el.text or ''

        if 'Cryptic Crossword' not in title:
            continue

        # Find the alternate link (the puzzle HTML page)
        for link in entry.findall(f'{{{ATOM_NS}}}link'):
            rel = link.get('rel', '')
            href = link.get('href', '')
            if rel == 'alternate' and href.endswith('-puzzle.html'):
                # href is relative like ../../pugpig_page/puzzler-puzzles/{id}/{ts}-puzzle.html
                # Convert to absolute URL
                if href.startswith('../../'):
                    puzzle_url = f"{EDITION_BASE}/{href[6:]}"
                elif href.startswith('http'):
                    puzzle_url = href
                else:
                    puzzle_url = f"{EDITION_BASE}/{href}"
                return puzzle_url, title

    return None, None


def fetch_puzzle_json(puzzle_url):
    """Fetch the puzzle HTML page and extract the inline JSON data.

    Returns the parsed JSON dict or None on failure.
    """
    try:
        resp = requests.get(puzzle_url, timeout=30)
        if resp.status_code != 200:
            print(f"  Puzzle page HTTP {resp.status_code}: {puzzle_url}")
            return None
    except Exception as e:
        print(f"  Error fetching puzzle page: {e}")
        return None

    # Server sends UTF-8 but claims text/html without charset,
    # causing requests to default to ISO-8859-1 and mangle unicode
    resp.encoding = 'utf-8'
    html = resp.text

    # Extract options.starting_puzzle = {...}
    match = re.search(r'options\.starting_puzzle\s*=\s*(\{.+?\})\s*\n', html, re.DOTALL)
    if not match:
        print("  Could not find starting_puzzle JSON in HTML")
        return None

    try:
        return json.loads(match.group(1))
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
        return None


def compute_clue_numbers(items):
    """Compute crossword clue numbers from grid start positions.

    Standard crossword numbering: collect all unique start positions,
    sort numerically, and assign sequential numbers 1, 2, 3...

    Returns a dict mapping (start, dir) -> clue_number.
    """
    unique_starts = sorted(set(item['start'] for item in items))
    start_to_number = {s: i + 1 for i, s in enumerate(unique_starts)}
    return start_to_number


def build_grid_solution(items, rows, cols):
    """Build a flat solution string from puzzle items.

    Same format as Guardian grids in puzzle_grids table.
    Returns solution string or None if no data.
    """
    grid = [[' '] * cols for _ in range(rows)]
    has_any = False

    for item in items:
        answer = item.get('answer', '')
        if not answer:
            continue
        start = item['start']
        direction = item['dir']  # 0 = across, 1 = down

        row = (start - 1) // cols
        col = (start - 1) % cols

        for i, ch in enumerate(answer):
            if direction == 0:  # across
                c = col + i
                if c < cols:
                    grid[row][c] = ch
                    has_any = True
            else:  # down
                r = row + i
                if r < rows:
                    grid[r][col] = ch
                    has_any = True

    if not has_any:
        return None

    return ''.join(''.join(r) for r in grid)


def extract_puzzle_number(title):
    """Extract puzzle number from title like 'Cryptic Crossword No. 12,166'.

    Also handles 'Cryptic Crossword 11,083' (without 'No.').
    Returns the number as a string (without commas) or None.
    """
    match = re.search(r'(?:No\.?\s*|Crossword\s+)([\d,]+)', title)
    if match:
        return match.group(1).replace(',', '')
    return None


def parse_puzzle(puzzle_json, puzzle_date, puzzle_title):
    """Parse the puzzle JSON into a list of clue dicts and grid data.

    Returns (clues, grid_solution, grid_rows, grid_cols, puzzle_number).
    """
    game_data = puzzle_json.get('game_data', {})
    items = game_data.get('items', [])
    rows = game_data.get('rows', 15)
    cols = game_data.get('cols', 15)
    author = game_data.get('author', '')

    puzzle_number = extract_puzzle_number(puzzle_title or puzzle_json.get('name', ''))

    if not items:
        return [], None, rows, cols, puzzle_number

    start_to_number = compute_clue_numbers(items)
    grid_solution = build_grid_solution(items, rows, cols)

    clues = []
    for item in items:
        clue_text_raw = item.get('clue', '')
        answer = item.get('answer', '').upper()
        direction = 'across' if item['dir'] == 0 else 'down'
        clue_number = start_to_number[item['start']]

        # Separate enumeration from clue text
        enum_match = re.search(r'\(([0-9,\-\s]+)\)\s*$', clue_text_raw)
        enumeration = enum_match.group(1).strip() if enum_match else None
        clue_text = re.sub(r'\s*\([0-9,\-\s]+\)\s*$', '', clue_text_raw).strip()

        clues.append({
            'clue_number': str(clue_number),
            'direction': direction,
            'clue_text': clue_text,
            'enumeration': enumeration,
            'answer': answer,
            'author': author,
        })

    return clues, grid_solution, rows, cols, puzzle_number


def save_to_database(clues, puzzle_date, puzzle_number, grid_solution, grid_rows, grid_cols,
                     source_url=None):
    """Save clues to clues table and grid to puzzle_grids.

    Returns count of clues saved.
    """
    if not clues or not puzzle_number:
        return 0

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    date_str = puzzle_date.isoformat()
    clue_count = 0

    for clue in clues:
        cursor.execute("""
            INSERT OR IGNORE INTO clues
            (source, publication_date, puzzle_number, clue_number, direction,
             clue_text, enumeration, answer)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            'independent',
            date_str,
            puzzle_number,
            clue['clue_number'],
            clue['direction'],
            clue['clue_text'],
            clue['enumeration'],
            clue['answer'],
        ))
        clue_count += 1

    # Save grid solution
    if grid_solution and len(grid_solution) == grid_rows * grid_cols:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS puzzle_grids (
                source TEXT NOT NULL,
                puzzle_number TEXT NOT NULL,
                solution TEXT,
                grid_rows INTEGER NOT NULL DEFAULT 15,
                grid_cols INTEGER NOT NULL DEFAULT 15,
                UNIQUE(source, puzzle_number)
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
        """, ('independent', puzzle_number, grid_solution, grid_rows, grid_cols, source_url))

    conn.commit()
    conn.close()
    return clue_count


def scrape_date(puzzle_date, dry_run=False):
    """Scrape the cryptic crossword for a given date.

    Returns (puzzle_number, clue_count) or (None, 0) on failure.
    """
    date_str = puzzle_date.strftime('%d %B %Y')
    print(f"  {date_str}: ", end="")

    # Step 1: Fetch content.xml
    xml_root = fetch_content_xml(puzzle_date)
    if xml_root is None:
        print("no edition found")
        return None, 0

    # Step 2: Find cryptic crossword entry
    puzzle_url, puzzle_title = find_cryptic_puzzle_url(xml_root)
    if not puzzle_url:
        print("no cryptic crossword in edition")
        return None, 0

    # Step 3: Fetch puzzle HTML and extract JSON
    puzzle_json = fetch_puzzle_json(puzzle_url)
    if not puzzle_json:
        print("failed to extract puzzle data")
        return None, 0

    # Step 4: Parse into clues + grid
    clues, grid_solution, grid_rows, grid_cols, puzzle_number = parse_puzzle(
        puzzle_json, puzzle_date, puzzle_title
    )

    if not clues:
        print("no clues found in puzzle data")
        return None, 0

    # Count across/down
    across = sum(1 for c in clues if c['direction'] == 'across')
    down = sum(1 for c in clues if c['direction'] == 'down')
    grid_tag = f", grid {grid_rows}x{grid_cols}" if grid_solution else ""

    if dry_run:
        print(f"#{puzzle_number} — {len(clues)} clues ({across}A + {down}D){grid_tag} [DRY RUN]")
        return puzzle_number, len(clues)

    # Step 5: Save to database
    clue_count = save_to_database(
        clues, puzzle_date, puzzle_number, grid_solution, grid_rows, grid_cols,
        source_url=puzzle_url
    )
    print(f"#{puzzle_number} — {clue_count} clues ({across}A + {down}D){grid_tag}")
    return puzzle_number, clue_count


def scrape_today(dry_run=False):
    """Scrape today's cryptic crossword."""
    today = date.today()
    print(f"Independent Edition Scraper — {today.strftime('%A %d %B %Y')}")
    print(f"Database: {DB_PATH}")
    puzzle_number, count = scrape_date(today, dry_run=dry_run)
    if count > 0:
        print(f"\nDone: puzzle #{puzzle_number}, {count} clues saved")
    else:
        print("\nNo puzzle found for today")
    return count


def main():
    dry_run = '--dry-run' in sys.argv

    if dry_run:
        print("[DRY RUN MODE -- no database writes]\n")

    # --date YYYY-MM-DD
    target_date = None
    for i, arg in enumerate(sys.argv):
        if arg == '--date' and i + 1 < len(sys.argv):
            target_date = date.fromisoformat(sys.argv[i + 1])

    if target_date:
        print(f"Independent Edition Scraper -- {target_date.strftime('%A %d %B %Y')}")
        print(f"Database: {DB_PATH}")
        _, count = scrape_date(target_date, dry_run=dry_run)
    else:
        count = scrape_today(dry_run=dry_run)

    # Exit with error if no clues found — orchestrator uses exit code
    if count == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
