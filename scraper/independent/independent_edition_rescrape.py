#!/usr/bin/env python3
"""Independent Edition Re-scrape

Deletes existing clues for specified dates and re-inserts from the edition
archive with correct clue numbers, direction, and grid data.

Also handles inserting brand new puzzles not yet in the DB.

Usage:
    python independent_edition_rescrape.py --failures    # Re-scrape from backfill_failures.txt
    python independent_edition_rescrape.py --new         # Scrape new puzzles (June 2021 -> DB start)
    python independent_edition_rescrape.py --all         # Both failures + new
    python independent_edition_rescrape.py --dry-run     # Preview without DB writes

Input: backfill_failures.txt (for --failures mode)
"""

import os
import re
import sqlite3
import sys
import time
from datetime import date, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from independent_edition import (
    REQUEST_DELAY,
    content_xml_url,
    fetch_content_xml,
    find_cryptic_puzzle_url,
    fetch_puzzle_json,
    parse_puzzle,
    save_to_database,
)

DB_PATH = os.getenv('DB_PATH',
                    r"C:\Users\shute\PycharmProjects\cryptic_solver_V2\data\clues_master.db")

EARLIEST_DATE = date(2021, 6, 11)
FAILURES_FILE = Path(__file__).resolve().parent / 'backfill_failures.txt'


def load_failure_dates():
    """Load dates from backfill_failures.txt."""
    if not FAILURES_FILE.exists():
        print(f"Failures file not found: {FAILURES_FILE}")
        return []

    dates = []
    with open(FAILURES_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Format: 2022-12-08 | #11281 | partial | 29/30
            parts = line.split('|')
            if len(parts) >= 2:
                date_str = parts[0].strip()
                try:
                    dates.append(date.fromisoformat(date_str))
                except ValueError:
                    continue
    return dates


def get_existing_dates():
    """Get all Independent publication dates in the DB."""
    if not Path(DB_PATH).exists():
        return set()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT publication_date FROM clues
        WHERE source = 'independent'
    """)
    dates = set(row[0] for row in cursor.fetchall())
    conn.close()
    return dates


def find_new_dates():
    """Find dates available in edition archive but not in DB."""
    existing = get_existing_dates()
    new_dates = []

    # Check each date from earliest to the start of DB coverage
    # Also check gaps within DB coverage
    d = EARLIEST_DATE
    today = date.today()

    while d <= today:
        if d.isoformat() not in existing:
            new_dates.append(d)
        d += timedelta(days=1)

    return new_dates


def scrape_and_insert(puzzle_date, dry_run=False, is_rescrape=False):
    """Scrape a puzzle from edition and insert into DB.

    If is_rescrape=True, deletes existing rows first.
    Returns (puzzle_number, clue_count) or (None, 0) on failure.
    """
    xml_root = fetch_content_xml(puzzle_date)
    if xml_root is None:
        return None, 0

    puzzle_url, puzzle_title = find_cryptic_puzzle_url(xml_root)
    if not puzzle_url:
        return None, 0

    puzzle_json = fetch_puzzle_json(puzzle_url)
    if not puzzle_json:
        return None, 0

    clues, grid_solution, grid_rows, grid_cols, puzzle_number = parse_puzzle(
        puzzle_json, puzzle_date, puzzle_title
    )
    if not clues:
        return None, 0

    across = sum(1 for c in clues if c['direction'] == 'across')
    down = sum(1 for c in clues if c['direction'] == 'down')
    grid_tag = f", grid {grid_rows}x{grid_cols}" if grid_solution else ""

    if dry_run:
        action = "RESCRAPE" if is_rescrape else "NEW"
        print(f"[DRY RUN {action}] #{puzzle_number} -- {len(clues)} clues ({across}A + {down}D){grid_tag}")
        return puzzle_number, len(clues)

    if is_rescrape:
        # Delete existing rows for this date
        conn = sqlite3.connect(DB_PATH)
        conn.execute("""
            DELETE FROM clues
            WHERE source = 'independent' AND publication_date = ?
        """, (puzzle_date.isoformat(),))
        conn.commit()
        conn.close()

    count = save_to_database(
        clues, puzzle_date, puzzle_number, grid_solution, grid_rows, grid_cols
    )

    action = "RESCRAPE" if is_rescrape else "NEW"
    print(f"[{action}] #{puzzle_number} -- {count} clues ({across}A + {down}D){grid_tag}")
    return puzzle_number, count


def main():
    dry_run = '--dry-run' in sys.argv
    do_failures = '--failures' in sys.argv or '--all' in sys.argv
    do_new = '--new' in sys.argv or '--all' in sys.argv

    if not do_failures and not do_new:
        print("Usage: specify --failures, --new, or --all")
        print("  --failures  Re-scrape puzzles from backfill_failures.txt")
        print("  --new       Scrape new puzzles not in DB")
        print("  --all       Both")
        print("  --dry-run   Preview without DB writes")
        return

    mode = "DRY RUN" if dry_run else "EXECUTING"
    print("=" * 70)
    print(f"INDEPENDENT EDITION RE-SCRAPE [{mode}]")
    print("=" * 70)
    print(f"Database: {DB_PATH}")
    print()

    rescrape_ok = 0
    rescrape_fail = 0
    new_ok = 0
    new_fail = 0
    new_skip = 0

    # Phase 1: Re-scrape failures
    if do_failures:
        failure_dates = load_failure_dates()
        print(f"PHASE 1: Re-scraping {len(failure_dates)} failure puzzles")
        print("-" * 50)

        for i, puzzle_date in enumerate(failure_dates):
            label = puzzle_date.strftime('%a %d %b %Y')
            print(f"  [{i+1}/{len(failure_dates)}] {label}: ", end="", flush=True)

            pnum, count = scrape_and_insert(puzzle_date, dry_run=dry_run, is_rescrape=True)
            if count > 0:
                rescrape_ok += 1
            else:
                print("edition not available")
                rescrape_fail += 1

            time.sleep(REQUEST_DELAY)

        print(f"\n  Re-scrape: {rescrape_ok} OK, {rescrape_fail} failed")
        print()

    # Phase 2: New puzzles
    if do_new:
        print("PHASE 2: Finding new puzzles...")
        new_dates = find_new_dates()
        print(f"  {len(new_dates)} dates not in DB")
        print("-" * 50)

        for i, puzzle_date in enumerate(new_dates):
            label = puzzle_date.strftime('%a %d %b %Y')
            print(f"  [{i+1}/{len(new_dates)}] {label}: ", end="", flush=True)

            # Check edition exists (HEAD request first to avoid slow fetches)
            import requests
            url = content_xml_url(puzzle_date)
            try:
                resp = requests.head(url, timeout=10)
                if resp.status_code != 200:
                    print("no edition")
                    new_skip += 1
                    time.sleep(0.3)
                    continue
            except Exception:
                print("error")
                new_skip += 1
                time.sleep(0.3)
                continue

            pnum, count = scrape_and_insert(puzzle_date, dry_run=dry_run, is_rescrape=False)
            if count > 0:
                new_ok += 1
            else:
                print("no cryptic puzzle")
                new_fail += 1

            time.sleep(REQUEST_DELAY)

        print(f"\n  New puzzles: {new_ok} OK, {new_fail} failed, {new_skip} no edition")
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if do_failures:
        print(f"  Re-scraped: {rescrape_ok} OK, {rescrape_fail} failed")
    if do_new:
        print(f"  New:        {new_ok} OK, {new_fail} failed, {new_skip} skipped")
    total = rescrape_ok + new_ok
    print(f"  Total puzzles added/updated: {total}")
    if dry_run:
        print("\n  This was a DRY RUN. No database changes were made.")


if __name__ == "__main__":
    main()
