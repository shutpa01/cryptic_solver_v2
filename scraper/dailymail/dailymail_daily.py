#!/usr/bin/env python3
"""Daily Mail Crossword Scraper

Fetches cryptic and quick crossword clues from the Daily Mail bundle API.
No login required — the API is unauthenticated.

API pattern: https://api.mailplus.co.uk/puzzles/mail-plus/data/YYYY-MM-DD/bundle.json

Game IDs of interest:
  6    - Daily Mail Cryptic Crossword (15x15, prize/competition)
  1034 - Quick Crossword (15x15)
  1033 - Quick Crossword (11x11)
  1020 - General Knowledge Crossword (15x15)
  31   - Pitcherwits (11x11, picture-based cryptic)
  12   - Quick Crossword (12x12)

We scrape gameId 6 (cryptic) by default. Use --all to include quick crosswords.
"""

import json
import os
import re
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

SCRIPT_DIR = Path(__file__).parent
DB_PATH = os.getenv('DB_PATH',
                     r"C:\Users\shute\PycharmProjects\cryptic_solver_V2\data\clues_master.db")

API_BASE = "https://api.mailplus.co.uk/puzzles/mail-plus/data"

# Paper puzzle numbering: weekdays (Mon-Fri) only, anchored to a known date.
# The API uses uniqueId which is unrelated to the printed puzzle number.
# Christmas Day behaviour varies by year — check and update XMAS_NUMBERED
# at the end of each December.
PUZZLE_NUMBER_ANCHOR_DATE = date(2024, 12, 26)
PUZZLE_NUMBER_ANCHOR_NUM = 17519
# Years where Christmas Day (if a weekday) WAS numbered (online puzzle got a number)
XMAS_NUMBERED = {2024}


def calculate_puzzle_number(target_date: date) -> int | None:
    """Calculate the printed newspaper puzzle number for a weekday date.

    Returns None for weekends and unnumbered Christmas Days.
    """
    if target_date.weekday() >= 5:
        return None
    if (target_date.month == 12 and target_date.day == 25
            and target_date.year not in XMAS_NUMBERED):
        return None

    anchor = PUZZLE_NUMBER_ANCHOR_DATE
    anchor_num = PUZZLE_NUMBER_ANCHOR_NUM

    if target_date == anchor:
        return anchor_num

    step = 1 if target_date > anchor else -1
    count = 0
    d = anchor + timedelta(days=step)
    while (step == 1 and d <= target_date) or (step == -1 and d >= target_date):
        if d.weekday() < 5:
            if d.month == 12 and d.day == 25 and d.year not in XMAS_NUMBERED:
                pass  # skip unnumbered Christmas
            else:
                count += 1
        d += timedelta(days=step)

    return anchor_num + (count * step)


# Game IDs to scrape, mapped to our source names
GAME_CONFIG = {
    6:    {'source': 'dailymail', 'type': 'cryptic', 'label': 'Daily Mail Cryptic'},
    1034: {'source': 'dailymail-quick', 'type': 'quick', 'label': 'Daily Mail Quick (15x15)'},
    1033: {'source': 'dailymail-quick', 'type': 'quick', 'label': 'Daily Mail Quick (11x11)'},
    1020: {'source': 'dailymail-quick', 'type': 'general-knowledge', 'label': 'Daily Mail General Knowledge'},
}

# Default: only scrape the cryptic
DEFAULT_GAME_IDS = [6]
ALL_GAME_IDS = [6, 1034, 1033, 1020]


def fetch_bundle(target_date: date) -> dict | None:
    """Fetch the daily puzzle bundle JSON for a given date."""
    date_str = target_date.strftime('%Y-%m-%d')
    url = f"{API_BASE}/{date_str}/bundle.json"
    print(f"  Fetching: {url}")

    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 404:
            print(f"  No bundle for {date_str} (404)")
            return None
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"  Error fetching bundle: {e}")
        return None


def extract_crossword_clues(game_data: dict, game_id: int) -> tuple[list, list]:
    """Extract across and down clues from a crossword game entry."""
    data = game_data.get('data', {})
    across = []
    down = []

    for clue in data.get('hor', []):
        across.append({
            'number': str(clue['nb']),
            'clue': clue['question'],
            'answer': clue['answer'],
            'enumeration': clue.get('len', ''),
        })

    for clue in data.get('ver', []):
        down.append({
            'number': str(clue['nb']),
            'clue': clue['question'],
            'answer': clue['answer'],
            'enumeration': clue.get('len', ''),
        })

    return across, down


def puzzle_already_fetched(source: str, puzzle_number: str) -> bool:
    """Check if puzzle is already in the clues table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM clues WHERE source = ? AND puzzle_number = ?",
        (source, puzzle_number)
    )
    count = cursor.fetchone()[0]
    conn.close()
    return count > 0


def save_clues(source: str, puzzle_number: str, pub_date: str,
               across: list, down: list) -> int:
    """Save clues to the clues table. Returns number of rows inserted."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    inserted = 0

    for direction, clue_list in [('across', across), ('down', down)]:
        for clue in clue_list:
            cursor.execute("""
                INSERT INTO clues
                (source, puzzle_number, publication_date, clue_number, direction,
                 clue_text, enumeration, answer)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source, puzzle_number, clue_number, direction, publication_date)
                DO UPDATE SET answer = excluded.answer
                WHERE answer IS NULL OR answer = ''
            """, (
                source,
                puzzle_number,
                pub_date,
                clue['number'],
                direction,
                clue['clue'],
                clue['enumeration'],
                clue['answer'],
            ))
            if cursor.rowcount > 0:
                inserted += 1

    conn.commit()
    conn.close()
    return inserted


def build_grid_solution(game_data):
    """Build a flat solution string from DM clue positions and answers.
    Returns (solution_string, rows, cols) or (None, 15, 15)."""
    data = game_data.get('data', {})
    rows = int(data.get('rows', 15))
    cols = int(data.get('cols', 15))

    grid = [[' '] * cols for _ in range(rows)]
    has_any = False

    for clue in data.get('hor', []):
        r_idx = clue['r'] - 1
        c_idx = clue['c'] - 1
        ans = clue.get('answer', '').replace(' ', '')
        for i, ch in enumerate(ans):
            if 0 <= c_idx + i < cols:
                grid[r_idx][c_idx + i] = ch.upper()
                has_any = True

    for clue in data.get('ver', []):
        r_idx = clue['r'] - 1
        c_idx = clue['c'] - 1
        ans = clue.get('answer', '').replace(' ', '')
        for i, ch in enumerate(ans):
            if 0 <= r_idx + i < rows:
                grid[r_idx + i][c_idx] = ch.upper()
                has_any = True

    if not has_any:
        return None, rows, cols

    return ''.join(''.join(row) for row in grid), rows, cols


def save_grid(source: str, puzzle_number: str, game_data: dict, source_url=None):
    """Save grid data and solution to puzzle_grids table."""
    data = game_data.get('data', {})
    rows = int(data.get('rows', 15))
    cols = int(data.get('cols', 15))

    solution, grid_rows, grid_cols = build_grid_solution(game_data)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

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

    cursor.execute("""
        INSERT INTO puzzle_grids
        (source, puzzle_number, solution, grid_rows, grid_cols, api_folder, api_type, api_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source, puzzle_number) DO UPDATE SET
            solution = COALESCE(excluded.solution, puzzle_grids.solution)
    """, (
        source, puzzle_number, solution, rows, cols,
        source_url or 'dailymail', 'bundle', str(game_data.get('uniqueId', '')),
    ))

    conn.commit()
    conn.close()


def scrape_date(target_date: date, game_ids: list[int]) -> dict:
    """Scrape all target crosswords for a single date."""
    stats = {'fetched': 0, 'skipped': 0, 'failed': 0}
    date_str = target_date.strftime('%Y-%m-%d')

    bundle = fetch_bundle(target_date)
    if not bundle:
        stats['failed'] += len(game_ids)
        return stats

    # Index games by gameId
    games_by_id = {g['gameId']: g for g in bundle.get('games', [])}

    for game_id in game_ids:
        config = GAME_CONFIG[game_id]
        label = config['label']

        if game_id not in games_by_id:
            print(f"  {label}: not found in bundle")
            stats['failed'] += 1
            continue

        game = games_by_id[game_id]
        unique_id = str(game.get('uniqueId', game_id))
        source = config['source']

        # Calculate printed puzzle number for weekday cryptics
        if source == 'dailymail':
            paper_num = calculate_puzzle_number(target_date)
            if paper_num is None:
                print(f"  {label}: {date_str} is not a numbered puzzle day — skipping")
                stats['skipped'] += 1
                continue
            puzzle_number = str(paper_num)
        else:
            puzzle_number = unique_id

        # Check for hor/ver clues (crossword format)
        data = game.get('data', {})
        if 'hor' not in data and 'ver' not in data:
            print(f"  {label}: no crossword clues (not a crossword game)")
            stats['failed'] += 1
            continue

        # Check if already fetched
        if puzzle_already_fetched(source, puzzle_number):
            print(f"  {label} #{puzzle_number}: already fetched — skipping")
            stats['skipped'] += 1
            continue

        across, down = extract_crossword_clues(game, game_id)
        total_clues = len(across) + len(down)

        if total_clues == 0:
            print(f"  {label}: no clues extracted")
            stats['failed'] += 1
            continue

        inserted = save_clues(source, puzzle_number, date_str, across, down)
        bundle_url = f"{API_BASE}/{date_str}/bundle.json"
        save_grid(source, puzzle_number, game, source_url=bundle_url)

        print(f"  {label} #{puzzle_number}: {len(across)}A + {len(down)}D = {total_clues} clues saved")
        stats['fetched'] += 1

    return stats


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Daily Mail Crossword Scraper')
    parser.add_argument('--date', type=str, default=None,
                        help='Date to scrape (YYYY-MM-DD). Default: today')
    parser.add_argument('--backfill', type=int, default=0,
                        help='Number of days to backfill (e.g. --backfill 30)')
    parser.add_argument('--all', action='store_true',
                        help='Scrape all crossword types, not just cryptic')
    args = parser.parse_args()

    game_ids = ALL_GAME_IDS if args.all else DEFAULT_GAME_IDS

    print("=" * 60)
    print("DAILY MAIL CROSSWORD SCRAPER")
    print(f"Database: {DB_PATH}")
    print(f"Games: {', '.join(GAME_CONFIG[g]['label'] for g in game_ids)}")
    print("=" * 60)

    if args.date:
        start_date = date.fromisoformat(args.date)
        dates = [start_date]
    elif args.backfill > 0:
        today = date.today()
        dates = [today - timedelta(days=i) for i in range(args.backfill)]
        print(f"Backfilling {args.backfill} days: {dates[-1]} to {dates[0]}")
    else:
        dates = [date.today()]

    totals = {'fetched': 0, 'skipped': 0, 'failed': 0}

    for target_date in dates:
        print(f"\n--- {target_date.strftime('%A, %d %B %Y')} ---")
        stats = scrape_date(target_date, game_ids)
        for k in totals:
            totals[k] += stats[k]

    print(f"\n{'=' * 60}")
    print(f"DONE — Fetched: {totals['fetched']}, "
          f"Skipped: {totals['skipped']}, Failed: {totals['failed']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
